"""LangGraph scheduler node and routing helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .action_mask import compute_action_mask
from .actions import SchedulerAction, parse_action
from .policy import PolicyDecision, SchedulerContext, SchedulerPolicy
from .state_serializer import serialize_scheduler_state

SCHEDULER_NODE_NAME = "Scheduler"


class SchedulerFallbackError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def progress_signature(state: Mapping[str, Any]) -> str:
    """Return a compact comparable signature of orchestration progress."""

    debate = state.get("investment_debate_state") or {}
    risk = state.get("risk_debate_state") or {}
    values = (
        len(str(state.get("market_report") or "")),
        len(str(state.get("sentiment_report") or "")),
        len(str(state.get("news_report") or "")),
        len(str(state.get("fundamentals_report") or "")),
        int(debate.get("count") or 0),
        len(str(state.get("investment_plan") or "")),
        len(str(state.get("trader_investment_plan") or "")),
        int(risk.get("count") or 0),
        len(str(state.get("final_trade_decision") or "")),
    )
    return "|".join(str(value) for value in values)


class SchedulerNode:
    def __init__(
        self,
        policy: SchedulerPolicy,
        selected_analysts: Sequence[str],
        *,
        max_steps: int = 16,
        max_debate_rounds: int = 1,
        max_risk_rounds: int = 1,
        max_decision_attempts: int = 2,
        on_decision: Callable[[SchedulerContext, PolicyDecision], None] | None = None,
    ):
        self.policy = policy
        self.selected_analysts = tuple(selected_analysts)
        self.max_steps = max_steps
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_rounds = max_risk_rounds
        self.max_decision_attempts = max_decision_attempts
        self.on_decision = on_decision

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        step = int(state.get("scheduler_step") or 0)
        current_signature = progress_signature(state)
        previous_signature = str(state.get("scheduler_last_state_signature") or "")
        previous_no_progress = int(state.get("scheduler_no_progress_count") or 0)
        no_progress = previous_no_progress + 1 if previous_signature == current_signature else 0
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
            raise SchedulerFallbackError(mask.reason or "no_legal_scheduler_action")

        working_state = dict(state)
        working_state["scheduler_no_progress_count"] = no_progress
        serialized = serialize_scheduler_state(
            working_state,
            mask.valid_actions,
            step=step,
            max_steps=self.max_steps,
        )
        context = SchedulerContext(
            state=working_state,
            serialized_state=serialized,
            valid_actions=mask.valid_actions,
            selected_analysts=self.selected_analysts,
            step=step,
        )
        decision = None
        last_error: Exception | None = None
        for _ in range(self.max_decision_attempts):
            try:
                candidate = self.policy.select_action(context)
                if candidate.action not in mask.valid_actions:
                    raise ValueError(f"masked_policy_action:{candidate.action.value}")
                decision = candidate
                break
            except (TypeError, ValueError) as exc:
                last_error = exc
        if decision is None:
            raise SchedulerFallbackError(f"invalid_policy_decision:{last_error}") from last_error

        if self.on_decision is not None:
            self.on_decision(context, decision)
        history = list(state.get("scheduler_history") or [])
        history.append(decision.action.value)
        return {
            "sender": SCHEDULER_NODE_NAME,
            "scheduler_action": decision.action.value,
            "scheduler_step": step + 1,
            "scheduler_history": history,
            "scheduler_valid_actions": [action.value for action in mask.valid_actions],
            "scheduler_policy_id": self.policy.policy_id,
            "scheduler_reason_code": decision.reason_code,
            "scheduler_reason": decision.reason,
            "scheduler_action_logprob": decision.logprob,
            "scheduler_last_state_signature": current_signature,
            "scheduler_no_progress_count": no_progress,
            "scheduler_agent_calls": int(state.get("scheduler_agent_calls") or 0)
            + (decision.action is not SchedulerAction.STOP),
        }


def route_scheduler_action(state: Mapping[str, Any]) -> str:
    """Return the canonical action token for a LangGraph conditional edge."""

    value = state.get("scheduler_action")
    if value is None:
        raise SchedulerFallbackError("scheduler_action_missing")
    return parse_action(value).value
