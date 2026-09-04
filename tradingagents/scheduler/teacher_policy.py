"""OpenRouter-backed Teacher policy with one state-preserving correction."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import requests

from .actions import SchedulerAction, parse_action
from .contracts import PolicyDecision, SchedulerContext
from .prompt import (
    TEACHER_PROMPT_VERSION,
    build_teacher_correction,
    build_teacher_messages,
    teacher_response_schema,
)
from .teacher_context import load_teacher_examples

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEACHER_MODEL = "z-ai/glm-5.3-flash"


class TeacherGatewayError(RuntimeError):
    """The Teacher request could not be completed."""


class TeacherOutputError(TeacherGatewayError):
    def __init__(self, reason: str, raw_output: object):
        super().__init__(reason)
        self.reason = reason
        self.raw_output = raw_output


class OpenRouterTeacherGateway:
    def __init__(
        self,
        *,
        model: str = DEFAULT_TEACHER_MODEL,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = OPENROUTER_BASE_URL,
        timeout_seconds: float = 90.0,
        temperature: float = 0.2,
        seed: int | None = None,
        transport: Callable[..., Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.seed = seed
        self._transport = transport or requests.post
        self._environ = os.environ if environ is None else environ

    def _api_key(self) -> str:
        value = self._environ.get(self.api_key_env, "").strip()
        if not value:
            raise TeacherGatewayError(
                f"Teacher requires environment variable {self.api_key_env}"
            )
        return value

    @staticmethod
    def _response_format(valid_actions: Sequence[SchedulerAction]) -> dict[str, object]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "scheduler_teacher_decision",
                "strict": True,
                "schema": teacher_response_schema(valid_actions),
            },
        }

    def request(
        self,
        messages: Sequence[Mapping[str, str]],
        valid_actions: Sequence[SchedulerAction],
    ) -> tuple[SchedulerAction, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "response_format": self._response_format(valid_actions),
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        try:
            response = self._transport(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", "unknown")
            message = self._http_error_message(exc.response)
            raise TeacherGatewayError(
                f"Teacher request failed with HTTP {status}: {message}"
            ) from exc
        except requests.RequestException as exc:
            raise TeacherGatewayError(f"Teacher request failed: {type(exc).__name__}") from exc

        try:
            raw_output = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TeacherOutputError("response_shape_invalid", body) from exc

        try:
            decision = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            if not isinstance(decision, Mapping) or set(decision) != {"action"}:
                raise ValueError("expected exactly one action field")
            action = parse_action(decision["action"])
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TeacherOutputError("action_json_invalid", raw_output) from exc
        if action not in valid_actions:
            raise TeacherOutputError("action_not_in_valid_actions", raw_output)
        return action, raw_output

    @staticmethod
    def _http_error_message(response: Any) -> str:
        try:
            body = response.json()
            message = body.get("error", {}).get("message")
            if message:
                return str(message)
        except (AttributeError, TypeError, ValueError):
            pass
        return "provider rejected the request"


class TeacherSchedulerPolicy:
    def __init__(
        self,
        gateway: OpenRouterTeacherGateway,
        *,
        positive_examples: Sequence[Mapping[str, object]] | None = None,
        failure_examples: Sequence[Mapping[str, object]] | None = None,
        context_dir: str | None = None,
    ):
        self.gateway = gateway
        if positive_examples is None or failure_examples is None:
            default_positive, default_failures = load_teacher_examples(context_dir)
            positive_examples = default_positive if positive_examples is None else positive_examples
            failure_examples = default_failures if failure_examples is None else failure_examples
        self.positive_examples = tuple(positive_examples)
        self.failure_examples = tuple(failure_examples)
        self.policy_id = f"teacher:{gateway.model}:{TEACHER_PROMPT_VERSION}"

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        messages = build_teacher_messages(
            context,
            positive_examples=self.positive_examples,
            failure_examples=self.failure_examples,
        )
        try:
            action, _ = self.gateway.request(messages, context.valid_actions)
            return self._decision(action, attempts=1, corrected=False)
        except TeacherOutputError as first_error:
            correction = build_teacher_correction(
                error_type=first_error.reason,
                previous_output=first_error.raw_output,
                valid_actions=context.valid_actions,
            )
            messages.extend(
                (
                    {"role": "assistant", "content": self._content(first_error.raw_output)},
                    {"role": "user", "content": correction},
                )
            )
            try:
                action, _ = self.gateway.request(messages, context.valid_actions)
            except TeacherOutputError as second_error:
                raise TeacherGatewayError(
                    f"Teacher correction failed: {second_error.reason}"
                ) from second_error
            return self._decision(action, attempts=2, corrected=True)

    @staticmethod
    def _content(value: object) -> str:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    def _decision(
        self, action: SchedulerAction, *, attempts: int, corrected: bool
    ) -> PolicyDecision:
        return PolicyDecision(
            action,
            policy_id=self.policy_id,
            decision_attempts=attempts,
            correction_succeeded=corrected,
            metadata={
                "teacher_model": self.gateway.model,
                "teacher_prompt_version": TEACHER_PROMPT_VERSION,
            },
        )
