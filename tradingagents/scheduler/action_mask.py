"""Deterministic execution constraints for central scheduler actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .actions import ANALYST_ACTION_BY_KEY, SchedulerAction, parse_action
from .registry import registry_for_analysts


@dataclass(frozen=True)
class ActionMask:
    valid_actions: tuple[SchedulerAction, ...]
    reason: str | None = None

    def allows(self, action: SchedulerAction | str) -> bool:
        return parse_action(action) in self.valid_actions


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _has_any_analyst_report(state: Mapping[str, Any]) -> bool:
    return any(
        _text(state.get(field))
        for field in (
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
        )
    )


def compute_action_mask(
    state: Mapping[str, Any],
    selected_analysts: tuple[str, ...] = (
        "market",
        "social",
        "news",
        "fundamentals",
    ),
    *,
    step: int = 0,
    max_steps: int = 16,
    max_debate_rounds: int = 1,
    max_risk_rounds: int = 1,
    last_action: SchedulerAction | str | None = None,
    no_progress_count: int = 0,
) -> ActionMask:
    """Compute executable actions without imposing the original fixed order."""

    if step < 0 or max_steps <= 0:
        raise ValueError("invalid scheduler step budget")
    if max_debate_rounds < 0 or max_risk_rounds < 0:
        raise ValueError("debate and risk rounds cannot be negative")
    registry_for_analysts(selected_analysts)

    if _text(state.get("final_trade_decision")):
        return ActionMask((SchedulerAction.STOP,))
    if step >= max_steps:
        return ActionMask((), "scheduler_max_steps_exceeded")

    valid: list[SchedulerAction] = []
    if not _text(state.get("investment_plan")):
        report_fields = {
            SchedulerAction.MARKET: "market_report",
            SchedulerAction.SENTIMENT: "sentiment_report",
            SchedulerAction.NEWS: "news_report",
            SchedulerAction.FUNDAMENTALS: "fundamentals_report",
        }
        for analyst_key in selected_analysts:
            action = ANALYST_ACTION_BY_KEY[analyst_key]
            if not _text(state.get(report_fields[action])):
                valid.append(action)

        debate = _mapping(state.get("investment_debate_state"))
        if _has_any_analyst_report(state):
            if int(debate.get("count") or 0) < 2 * max_debate_rounds:
                valid.extend((SchedulerAction.BULL, SchedulerAction.BEAR))
            if _text(debate.get("history")):
                valid.append(SchedulerAction.RESEARCH_MANAGER)
    elif not _text(state.get("trader_investment_plan")):
        valid.append(SchedulerAction.TRADER)
    else:
        risk = _mapping(state.get("risk_debate_state"))
        if int(risk.get("count") or 0) < 3 * max_risk_rounds:
            valid.extend(
                (
                    SchedulerAction.AGGRESSIVE,
                    SchedulerAction.CONSERVATIVE,
                    SchedulerAction.NEUTRAL,
                )
            )
        if _text(risk.get("history")):
            valid.append(SchedulerAction.PORTFOLIO_MANAGER)

    if last_action is not None and no_progress_count > 0:
        repeated = parse_action(last_action)
        valid = [action for action in valid if action is not repeated]

    unique = tuple(dict.fromkeys(valid))
    return ActionMask(unique, None if unique else "no_legal_scheduler_action")
