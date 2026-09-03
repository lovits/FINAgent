import json

import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import SchedulerContext
from tradingagents.scheduler.teacher_policy import (
    OpenRouterTeacherGateway,
    TeacherGatewayError,
    TeacherSchedulerPolicy,
)


class _Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content}}]}


def _context() -> SchedulerContext:
    return SchedulerContext(
        task_id="AAPL:2026-01-05",
        state={},
        serialized_state="<VALID_ACTIONS>\n[\"<ACT_MARKET>\", \"<ACT_NEWS>\"]",
        valid_actions=(SchedulerAction.MARKET, SchedulerAction.NEWS),
        selected_analysts=("market", "news"),
    )


def test_gateway_requires_environment_key() -> None:
    gateway = OpenRouterTeacherGateway(environ={}, transport=lambda **_: _Response("{}"))
    with pytest.raises(TeacherGatewayError, match="OPENROUTER_API_KEY"):
        gateway.request([], (SchedulerAction.MARKET,))


def test_teacher_returns_one_valid_action_and_structured_schema() -> None:
    calls = []

    def transport(*args, **kwargs):
        calls.append((args, kwargs))
        return _Response(json.dumps({"action": "<ACT_NEWS>"}))

    gateway = OpenRouterTeacherGateway(
        environ={"OPENROUTER_API_KEY": "test-only"}, transport=transport, seed=7
    )
    decision = TeacherSchedulerPolicy(gateway).select_action(_context())

    assert decision.action is SchedulerAction.NEWS
    assert decision.decision_attempts == 1
    payload = calls[0][1]["json"]
    assert payload["seed"] == 7
    schema = payload["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["action"]["enum"] == ["<ACT_MARKET>", "<ACT_NEWS>"]


def test_teacher_corrects_invalid_output_once_without_new_state() -> None:
    responses = iter(
        (
            _Response("I would call the trader"),
            _Response(json.dumps({"action": "<ACT_MARKET>"})),
        )
    )
    calls = []

    def transport(*args, **kwargs):
        calls.append(kwargs["json"])
        return next(responses)

    gateway = OpenRouterTeacherGateway(
        environ={"OPENROUTER_API_KEY": "test-only"}, transport=transport
    )
    decision = TeacherSchedulerPolicy(gateway).select_action(_context())

    assert decision.action is SchedulerAction.MARKET
    assert decision.decision_attempts == 2
    assert decision.correction_succeeded is True
    correction_messages = calls[1]["messages"]
    assert "CURRENT_STATE is unchanged" in correction_messages[-1]["content"]


def test_teacher_rejects_second_invalid_output() -> None:
    responses = iter((_Response("bad"), _Response('{"action":"<ACT_TRADER>"}')))
    gateway = OpenRouterTeacherGateway(
        environ={"OPENROUTER_API_KEY": "test-only"},
        transport=lambda *args, **kwargs: next(responses),
    )
    with pytest.raises(TeacherGatewayError, match="correction failed"):
        TeacherSchedulerPolicy(gateway).select_action(_context())
