import json

import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.agent_registry import registry_for_analysts
from tradingagents.scheduler.teacher_gateway import (
    DEFAULT_TEACHER_MODEL,
    OpenRouterTeacherGateway,
    TeacherGatewayError,
)
from tradingagents.scheduler.teacher_prompt import (
    TEACHER_PROMPT_VERSION,
    TeacherPromptInput,
    build_teacher_messages,
)


class _FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


def _prompt():
    registry = registry_for_analysts(("market", "news"))
    return build_teacher_messages(
        TeacherPromptInput(
            serialized_state='{"task":{"ticker":"NVDA"}}',
            valid_actions=(SchedulerAction.MARKET, SchedulerAction.NEWS),
            agent_specs=tuple(registry.values()),
            hard_rules=("Choose only a valid action.",),
            failure_examples=(
                {
                    "invalid_action": "<ACT_STOP>",
                    "corrected_actions": ["<ACT_MARKET>"],
                },
            ),
        )
    )


def test_teacher_prompt_is_versioned_and_grounded():
    messages = _prompt()

    assert f"TEACHER_PROMPT_VERSION={TEACHER_PROMPT_VERSION}" in messages[0]["content"]
    assert "<ACT_MARKET>" in messages[0]["content"]
    assert "VALID_ACTIONS" in messages[1]["content"]
    assert "OPENROUTER_API_KEY" not in json.dumps(messages)


def test_gateway_uses_environment_and_validates_action_without_network():
    captured = {}

    def transport(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "next_agent": "<ACT_NEWS>",
                                    "reason_code": "MISSING_NEWS",
                                    "reason": "Need event evidence.",
                                }
                            )
                        }
                    }
                ]
            }
        )

    gateway = OpenRouterTeacherGateway(
        transport=transport,
        environ={"OPENROUTER_API_KEY": "test-only-key"},
    )
    decision = gateway.select_action(
        _prompt(), (SchedulerAction.MARKET, SchedulerAction.NEWS)
    )

    assert decision.action is SchedulerAction.NEWS
    assert captured["url"].endswith("/chat/completions")
    assert captured["json"]["model"] == DEFAULT_TEACHER_MODEL
    assert captured["headers"]["Authorization"] == "Bearer test-only-key"
    assert "test-only-key" not in json.dumps(captured["json"])


def test_gateway_requires_key_and_rejects_invalid_response():
    no_key = OpenRouterTeacherGateway(transport=lambda *args, **kwargs: None, environ={})
    with pytest.raises(TeacherGatewayError, match="requires environment variable"):
        no_key.select_action(_prompt(), (SchedulerAction.MARKET,))

    invalid = OpenRouterTeacherGateway(
        transport=lambda *args, **kwargs: _FakeResponse({"choices": []}),
        environ={"OPENROUTER_API_KEY": "test-only-key"},
    )
    with pytest.raises(TeacherGatewayError, match="invalid response"):
        invalid.select_action(_prompt(), (SchedulerAction.MARKET,))
