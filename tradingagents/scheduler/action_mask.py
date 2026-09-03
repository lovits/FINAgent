"""Hard legality constraints for scheduler actions.

The mask excludes actions that cannot execute from the current state. It does
not encode the static workflow order; ordering remains a policy decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .actions import ANALYST_ACTION_BY_KEY, SchedulerAction, parse_action
from .agent_registry import registry_for_analysts


@dataclass(frozen=True)
class ActionMask:
    valid_actions: tuple[SchedulerAction, ...]
    reason: str | None = None

    def allows(self, action: SchedulerAction | str) -> bool:
        return parse_action(action) in self.valid_actions


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_analysis(state: Mapping[str, Any]) -> bool:
    return any(
        _has_text(state.get(field))
        for field in (
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
        )
    )


def compute_action_mask(
    state: Mapping[str, Any],
    selected_analysts: Sequence[str] = ("market", "social", "news", "fundamentals"),
    *,
    step: int = 0,
    max_steps: int = 16,
    max_debate_rounds: int = 1,
    max_risk_rounds: int = 1,
    last_action: SchedulerAction | str | None = None,
    no_progress_count: int = 0,
) -> ActionMask:
    """Compute executable actions without imposing the static sequence."""

    if _has_text(state.get("final_trade_decision")):
        return ActionMask((SchedulerAction.STOP,))
    if step >= max_steps:
        return ActionMask((), "scheduler_max_steps_exceeded")

    registry = registry_for_analysts(tuple(selected_analysts))
    valid: list[SchedulerAction] = []

    report_fields = {
        SchedulerAction.MARKET: "market_report",
        SchedulerAction.SENTIMENT: "sentiment_report",
        SchedulerAction.NEWS: "news_report",
        SchedulerAction.FUNDAMENTALS: "fundamentals_report",
    }
    for analyst_key in selected_analysts:
        action = ANALYST_ACTION_BY_KEY.get(analyst_key)
        if action in registry and not _has_text(state.get(report_fields[action])):
            valid.append(action)

    debate = _mapping(state.get("investment_debate_state"))
    debate_count = int(debate.get("count") or 0)
    debate_history = debate.get("history")
    if _has_analysis(state) and debate_count < 2 * max_debate_rounds:
        valid.extend((SchedulerAction.BULL, SchedulerAction.BEAR))
    if _has_text(debate_history):
        valid.append(SchedulerAction.RESEARCH_MANAGER)

    if _has_text(state.get("investment_plan")):
        valid.append(SchedulerAction.TRADER)

    risk = _mapping(state.get("risk_debate_state"))
    risk_count = int(risk.get("count") or 0)
    risk_history = risk.get("history")
    if _has_text(state.get("trader_investment_plan")):
        if risk_count < 3 * max_risk_rounds:
            valid.extend(
                (
                    SchedulerAction.AGGRESSIVE,
                    SchedulerAction.CONSERVATIVE,
                    SchedulerAction.NEUTRAL,
                )
            )
        if _has_text(risk_history):
            valid.append(SchedulerAction.PORTFOLIO_MANAGER)

    if last_action is not None and no_progress_count > 0:
        repeated = parse_action(last_action)
        valid = [action for action in valid if action is not repeated]

    deduplicated = tuple(dict.fromkeys(valid))
    reason = None if deduplicated else "no_legal_scheduler_action"
    return ActionMask(deduplicated, reason)
