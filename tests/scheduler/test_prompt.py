import json

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import SchedulerContext
from tradingagents.scheduler.prompt import (
    build_scheduler_input,
    build_teacher_correction,
    build_teacher_messages,
    teacher_response_schema,
)


def _state() -> dict:
    return {
        "messages": [{"tool": "raw data that must stay out"}],
        "company_of_interest": "AAPL",
        "asset_type": "stock",
        "instrument_context": "Apple Inc.; NASDAQ",
        "trade_date": "2026-01-05",
        "past_context": "Only decisions before 2026-01-05",
        "market_report": "M" * 2000,
        "sentiment_report": "",
        "news_report": "News evidence",
        "fundamentals_report": "",
        "investment_debate_state": {"history": "", "count": 0},
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {"history": "", "count": 0},
        "final_trade_decision": "",
    }


def _serialized() -> str:
    return build_scheduler_input(
        task_id="AAPL_2026-01-05_quiet_control",
        state=_state(),
        valid_actions=(SchedulerAction.SENTIMENT, SchedulerAction.BULL),
        selected_analysts=("market", "social", "news", "fundamentals"),
        history=(SchedulerAction.MARKET, SchedulerAction.NEWS),
        step=2,
        max_steps=16,
    )


def test_scheduler_input_keeps_complete_reports_and_excludes_raw_messages() -> None:
    prompt = _serialized()
    assert "M" * 2000 in prompt
    assert "raw data that must stay out" not in prompt
    assert "<ACT_SENTIMENT>" in prompt
    assert '"remaining_steps": 14' in prompt
    assert "tool_policy_owner" in prompt
    assert 'ORCHESTRATION_PROFILE version="multi-analyst-shallow-v1"' in prompt
    assert '"research_style": "shallow"' in prompt
    assert '"selected_analysts_are_required": true' in prompt
    assert "every analyst selected by the task input exactly once" in prompt


def test_scheduler_input_uses_one_profile_for_single_analyst_pool() -> None:
    prompt = build_scheduler_input(
        task_id="AAPL_2026-01-05_quiet_control",
        state=_state(),
        valid_actions=(SchedulerAction.MARKET,),
        selected_analysts=("market",),
    )

    assert '"selected_analyst_count": 1' in prompt
    assert '"selected_analysts": ["market"]' in prompt
    assert '"analyst_mode"' not in prompt
    assert '"research_style": "shallow"' in prompt


def test_teacher_schema_is_locked_to_valid_actions() -> None:
    schema = teacher_response_schema((SchedulerAction.NEWS, SchedulerAction.BULL))
    assert schema["required"] == ["action"]
    assert schema["additionalProperties"] is False
    enum = schema["properties"]["action"]["enum"]  # type: ignore[index]
    assert enum == ["<ACT_NEWS>", "<ACT_BULL>"]


def test_teacher_messages_include_prompt_and_machine_readable_schema() -> None:
    context = SchedulerContext(
        task_id="AAPL_2026-01-05_quiet_control",
        state=_state(),
        serialized_state=_serialized(),
        valid_actions=(SchedulerAction.SENTIMENT, SchedulerAction.BULL),
        selected_analysts=("market", "social", "news", "fundamentals"),
        history=(SchedulerAction.MARKET, SchedulerAction.NEWS),
        step=2,
    )
    messages = build_teacher_messages(
        context,
        positive_examples=({"state": "final ready", "action": "<ACT_STOP>"},),
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "<ACT_STOP>" in messages[0]["content"]
    assert "OUTPUT_SCHEMA=" in messages[1]["content"]
    schema_text = messages[1]["content"].split("OUTPUT_SCHEMA=", 1)[1]
    assert json.loads(schema_text)["required"] == ["action"]


def test_correction_preserves_state_and_allows_one_final_attempt() -> None:
    correction = build_teacher_correction(
        error_type="action_not_valid",
        previous_output={"action": "<ACT_TRADER>"},
        valid_actions=(SchedulerAction.MARKET, SchedulerAction.NEWS),
    )
    assert "CURRENT_STATE is unchanged" in correction
    assert "final correction attempt" in correction
    assert "<ACT_MARKET>" in correction
