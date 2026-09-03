"""Deterministic, bounded state representation shared by Teacher and Scheduler."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .actions import SchedulerAction, parse_action

STATE_SCHEMA_VERSION = "v1"


def _bounded_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def serialize_scheduler_state(
    state: Mapping[str, Any],
    valid_actions: Sequence[SchedulerAction | str],
    *,
    step: int,
    max_steps: int,
    report_char_limit: int = 600,
) -> str:
    """Serialize only orchestration-relevant state, excluding raw messages/tools."""

    debate = _mapping(state.get("investment_debate_state"))
    risk = _mapping(state.get("risk_debate_state"))
    history = state.get("scheduler_history") or []
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "task": {
            "ticker": state.get("company_of_interest", ""),
            "trade_date": state.get("trade_date", ""),
            "asset_type": state.get("asset_type", "stock"),
            "instrument_context": _bounded_text(state.get("instrument_context"), 300),
        },
        "reports": {
            "market": _bounded_text(state.get("market_report"), report_char_limit),
            "sentiment": _bounded_text(state.get("sentiment_report"), report_char_limit),
            "news": _bounded_text(state.get("news_report"), report_char_limit),
            "fundamentals": _bounded_text(
                state.get("fundamentals_report"), report_char_limit
            ),
        },
        "research": {
            "count": int(debate.get("count") or 0),
            "latest": _bounded_text(debate.get("current_response"), report_char_limit),
            "plan": _bounded_text(state.get("investment_plan"), report_char_limit),
        },
        "trade": {
            "proposal": _bounded_text(
                state.get("trader_investment_plan"), report_char_limit
            ),
        },
        "risk": {
            "count": int(risk.get("count") or 0),
            "latest_speaker": risk.get("latest_speaker", ""),
            "latest": _bounded_text(risk.get("history"), report_char_limit),
            "final_decision": _bounded_text(
                state.get("final_trade_decision"), report_char_limit
            ),
        },
        "execution": {
            "step": step,
            "max_steps": max_steps,
            "remaining_steps": max(0, max_steps - step),
            "history": [str(item) for item in history[-16:]],
            "agent_calls": int(state.get("scheduler_agent_calls") or 0),
            "tool_calls": int(state.get("scheduler_tool_calls") or 0),
            "estimated_tokens": int(state.get("scheduler_estimated_tokens") or 0),
        },
        "valid_actions": [parse_action(action).value for action in valid_actions],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
