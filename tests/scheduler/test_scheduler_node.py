import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import PolicyDecision
from tradingagents.scheduler.policy import CallableSchedulerPolicy
from tradingagents.scheduler.scheduler_node import (
    SchedulerNode,
    SchedulerRuntimeError,
    route_scheduler_action,
)


def _state() -> dict:
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2026-01-05",
        "asset_type": "stock",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {"history": "", "count": 0},
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {"history": "", "count": 0},
        "final_trade_decision": "",
    }


def test_scheduler_node_records_a_legal_decision() -> None:
    observed = []
    node = SchedulerNode(
        CallableSchedulerPolicy(lambda _: SchedulerAction.MARKET, policy_id="test-policy"),
        ("market", "news"),
        on_decision=lambda context, decision: observed.append((context, decision)),
    )

    update = node(_state())

    assert update["scheduler_action"] == "<ACT_MARKET>"
    assert update["scheduler_history"] == ["<ACT_MARKET>"]
    assert update["scheduler_step"] == 1
    assert update["scheduler_agent_calls"] == 1
    assert observed[0][0].serialized_state.endswith("<SCHEDULER_ACTION>\n")


def test_scheduler_node_exposes_decision_metadata_for_web_trace() -> None:
    node = SchedulerNode(
        CallableSchedulerPolicy(
            lambda _: PolicyDecision(
                SchedulerAction.MARKET,
                policy_id="test-policy",
                metadata={"usage": {"input_tokens": 25, "output_tokens": 1}},
            )
        ),
        ("market",),
    )

    update = node(_state())

    assert update["scheduler_decision_metadata"]["usage"] == {
        "input_tokens": 25,
        "output_tokens": 1,
    }


def test_scheduler_node_does_not_count_stop_as_agent_call() -> None:
    state = _state()
    state["final_trade_decision"] = "**Rating**: Hold"
    state["scheduler_agent_calls"] = 4
    node = SchedulerNode(
        CallableSchedulerPolicy(lambda _: SchedulerAction.STOP),
        ("market",),
    )

    update = node(state)
    assert update["scheduler_agent_calls"] == 4
    assert route_scheduler_action(update) == "<ACT_STOP>"


def test_scheduler_node_turns_policy_errors_into_runtime_errors() -> None:
    node = SchedulerNode(
        CallableSchedulerPolicy(lambda _: SchedulerAction.TRADER),
        ("market",),
    )
    with pytest.raises(SchedulerRuntimeError, match="invalid_policy_decision"):
        node(_state())


def test_scheduler_node_wraps_teacher_gateway_errors() -> None:
    def fail(_):
        raise RuntimeError("provider unavailable")

    node = SchedulerNode(CallableSchedulerPolicy(fail), ("market",))
    with pytest.raises(SchedulerRuntimeError, match="provider unavailable"):
        node(_state())


def test_route_requires_scheduler_action() -> None:
    with pytest.raises(SchedulerRuntimeError, match="scheduler_action_missing"):
        route_scheduler_action({})
