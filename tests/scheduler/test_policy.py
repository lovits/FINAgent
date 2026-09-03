import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import (
    CallableSchedulerPolicy,
    SchedulerContext,
    StaticSchedulerPolicy,
)


def _context(state, valid_actions, *, selected=("market",), step=0):
    return SchedulerContext(
        state=state,
        serialized_state="{}",
        valid_actions=tuple(valid_actions),
        selected_analysts=tuple(selected),
        step=step,
    )


def test_static_policy_reproduces_major_static_stages():
    policy = StaticSchedulerPolicy()

    initial = {"investment_debate_state": {}, "risk_debate_state": {}}
    decision = policy.select_action(_context(initial, [SchedulerAction.MARKET]))
    assert decision.action is SchedulerAction.MARKET

    after_analyst = {
        "market_report": "done",
        "investment_debate_state": {"count": 0, "current_response": ""},
        "risk_debate_state": {},
    }
    decision = policy.select_action(_context(after_analyst, [SchedulerAction.BULL]))
    assert decision.action is SchedulerAction.BULL

    final = {
        "market_report": "done",
        "investment_debate_state": {"count": 2, "current_response": "Bear"},
        "investment_plan": "plan",
        "trader_investment_plan": "Buy",
        "risk_debate_state": {"count": 3, "latest_speaker": "Neutral"},
        "final_trade_decision": "Overweight",
    }
    decision = policy.select_action(_context(final, [SchedulerAction.STOP]))
    assert decision.action is SchedulerAction.STOP


def test_callable_policy_rejects_masked_action():
    policy = CallableSchedulerPolicy(lambda _: SchedulerAction.TRADER)
    context = _context({}, [SchedulerAction.MARKET])

    with pytest.raises(ValueError, match="masked action"):
        policy.select_action(context)
