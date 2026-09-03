import json

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.state_serializer import (
    STATE_SCHEMA_VERSION,
    serialize_scheduler_state,
)


def test_serializer_is_deterministic_bounded_and_excludes_messages():
    state = {
        "company_of_interest": "NVDA",
        "trade_date": "2025-01-02",
        "asset_type": "stock",
        "market_report": " bullish   momentum " * 200,
        "messages": ["secret raw tool payload"],
        "investment_debate_state": {"count": 1, "current_response": "Bull case"},
        "risk_debate_state": {"count": 0},
        "scheduler_history": [SchedulerAction.MARKET.value],
    }

    first = serialize_scheduler_state(
        state,
        (SchedulerAction.NEWS, SchedulerAction.BEAR),
        step=1,
        max_steps=16,
        report_char_limit=80,
    )
    second = serialize_scheduler_state(
        state,
        (SchedulerAction.NEWS, SchedulerAction.BEAR),
        step=1,
        max_steps=16,
        report_char_limit=80,
    )

    assert first == second
    assert "secret raw tool payload" not in first
    payload = json.loads(first)
    assert payload["schema_version"] == STATE_SCHEMA_VERSION
    assert payload["execution"]["remaining_steps"] == 15
    assert payload["valid_actions"] == ["<ACT_NEWS>", "<ACT_BEAR>"]
    assert len(payload["reports"]["market"]) == 80
