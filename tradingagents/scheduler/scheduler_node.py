"""LangGraph node that delegates one Expert Agent decision to a scheduler policy."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .action_mask import compute_action_mask
from .actions import SchedulerAction, parse_action
from .contracts import PolicyDecision, SchedulerContext, SchedulerPolicy
from .prompt import build_scheduler_input

SCHEDULER_NODE_NAME = "Scheduler"


class SchedulerRuntimeError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def progress_signature(state: Mapping[str, Any]) -> str:
    """Return a stable signature of fields an Expert Agent is expected to change."""

    debate = state.get("investment_debate_state") or {}
    risk = state.get("risk_debate_state") or {}
    payload = {
        "market_report": state.get("market_report") or "",
        "sentiment_report": state.get("sentiment_report") or "",
        "news_report": state.get("news_report") or "",
        "fundamentals_report": state.get("fundamentals_report") or "",
        "research_count": debate.get("count", 0),
        "research_history": debate.get("history", ""),
        "investment_plan": state.get("investment_plan") or "",
        "trader_plan": state.get("trader_investment_plan") or "",
        "risk_count": risk.get("count", 0),
        "risk_history": risk.get("history", ""),
        "final_decision": state.get("final_trade_decision") or "",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


class SchedulerNode:
    def __init__(
        self,
        policy: SchedulerPolicy,
        selected_analysts: tuple[str, ...],
        *,
        max_steps: int = 16,
        max_debate_rounds: int = 1,
        max_risk_rounds: int = 1,
        on_decision: Callable[[SchedulerContext, PolicyDecision], None] | None = None,
    ):
        self.policy = policy
        self.selected_analysts = selected_analysts
        self.max_steps = max_steps
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_rounds = max_risk_rounds
        self.on_decision = on_decision

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        step = int(state.get("scheduler_step") or 0)
        history = tuple(state.get("scheduler_history") or ())
        current_signature = progress_signature(state)
        previous_signature = str(state.get("scheduler_last_state_signature") or "")
        previous_no_progress = int(state.get("scheduler_no_progress_count") or 0)
        no_progress = previous_no_progress + 1 if current_signature == previous_signature else 0
        last_action = state.get("scheduler_action")

        mask = compute_action_mask(
            state,
            self.selected_analysts,
            step=step,
            max_steps=self.max_steps,
            max_debate_rounds=self.max_debate_rounds,
            max_risk_rounds=self.max_risk_rounds,
            last_action=last_action,
            no_progress_count=no_progress,
        )
        if not mask.valid_actions:
            raise SchedulerRuntimeError(mask.reason or "no_legal_scheduler_action")

        task_id = str(
            state.get("scheduler_task_id")
            or f"{state.get('company_of_interest', 'unknown')}:{state.get('trade_date', 'unknown')}"
        )
        serialized = build_scheduler_input(
            task_id=task_id,
            state=state,
            valid_actions=mask.valid_actions,
            selected_analysts=self.selected_analysts,
            history=history,
            step=step,
            max_steps=self.max_steps,
            no_progress_count=no_progress,
        )
        context = SchedulerContext(
            task_id=task_id,
            state=state,
            serialized_state=serialized,
            valid_actions=mask.valid_actions,
            selected_analysts=self.selected_analysts,
            history=history,
            step=step,
            max_steps=self.max_steps,
            no_progress_count=no_progress,
        )
        try:
            decision = self.policy.select_action(context)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise SchedulerRuntimeError(f"invalid_policy_decision:{exc}") from exc
        if decision.action not in mask.valid_actions:
            raise SchedulerRuntimeError(f"masked_policy_action:{decision.action.value}")

        if self.on_decision is not None:
            self.on_decision(context, decision)
        next_history = [action.value for action in history]
        next_history.append(decision.action.value)
        return {
            "sender": SCHEDULER_NODE_NAME,
            "scheduler_action": decision.action.value,
            "scheduler_step": step + 1,
            "scheduler_history": next_history,
            "scheduler_valid_actions": [action.value for action in mask.valid_actions],
            "scheduler_policy_id": decision.policy_id,
            "scheduler_no_progress_count": no_progress,
            "scheduler_last_state_signature": current_signature,
            "scheduler_agent_calls": int(state.get("scheduler_agent_calls") or 0)
            + (decision.action is not SchedulerAction.STOP),
        }


def route_scheduler_action(state: Mapping[str, Any]) -> str:
    value = state.get("scheduler_action")
    if value is None:
        raise SchedulerRuntimeError("scheduler_action_missing")
    return parse_action(value).value
