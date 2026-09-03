"""OpenRouter Strong Teacher gateway with strict structured-output validation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import requests

from .actions import SchedulerAction, parse_action
from .policy import PolicyDecision
from .teacher_prompt import TEACHER_PROMPT_VERSION

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEACHER_MODEL = "google/gemini-3.8-flash"


class TeacherGatewayError(RuntimeError):
    """Raised when the Teacher request cannot produce a valid scheduler action."""


class OpenRouterTeacherGateway:
    def __init__(
        self,
        *,
        model: str = DEFAULT_TEACHER_MODEL,
        base_url: str = OPENROUTER_BASE_URL,
        api_key_env: str = "OPENROUTER_API_KEY",
        timeout_seconds: float = 60.0,
        transport: Callable[..., Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self._transport = transport or requests.post
        self._environ = environ if environ is not None else os.environ

    def _api_key(self) -> str:
        value = self._environ.get(self.api_key_env, "").strip()
        if not value:
            raise TeacherGatewayError(
                f"Strong Teacher requires environment variable {self.api_key_env}"
            )
        return value

    @staticmethod
    def _response_schema(valid_actions: Sequence[SchedulerAction]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "scheduler_teacher_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "next_agent": {
                            "type": "string",
                            "enum": [action.value for action in valid_actions],
                        },
                        "reason_code": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["next_agent", "reason_code", "reason"],
                    "additionalProperties": False,
                },
            },
        }

    def select_action(
        self,
        messages: Sequence[Mapping[str, str]],
        valid_actions: Sequence[SchedulerAction | str],
    ) -> PolicyDecision:
        parsed_valid = tuple(parse_action(action) for action in valid_actions)
        if not parsed_valid:
            raise TeacherGatewayError("Strong Teacher received no valid actions")
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.2,
            "response_format": self._response_schema(parsed_valid),
        }
        response = self._transport(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            decision = json.loads(content) if isinstance(content, str) else content
            action = parse_action(decision["next_agent"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TeacherGatewayError("Strong Teacher returned an invalid response") from exc
        except requests.RequestException as exc:
            raise TeacherGatewayError("Strong Teacher request failed") from exc

        if action not in parsed_valid:
            raise TeacherGatewayError(f"Strong Teacher selected masked action {action.value}")
        return PolicyDecision(
            action=action,
            reason_code=str(decision.get("reason_code", "")),
            reason=str(decision.get("reason", "")),
            metadata={
                "teacher_model": self.model,
                "teacher_prompt_version": TEACHER_PROMPT_VERSION,
            },
        )
