"""Scheduler policy interfaces and deterministic test policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .actions import ANALYST_ACTION_BY_KEY, SchedulerAction, parse_action


@dataclass(frozen=True)
class SchedulerContext:
    state: Mapping[str, Any]
    serialized_state: str
    valid_actions: tuple[SchedulerAction, ...]
    selected_analysts: tuple[str, ...]
    step: int


@dataclass(frozen=True)
class PolicyDecision:
    action: SchedulerAction
    reason_code: str = ""
    reason: str = ""
    logprob: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SchedulerPolicy(Protocol):
    policy_id: str

    def select_action(self, context: SchedulerContext) -> PolicyDecision: ...


class StaticSchedulerPolicy:
    """Mirror the original static path inside the learned graph for testing."""

    policy_id = "static-scheduler-v1"

    def __init__(self, *, max_debate_rounds: int = 1, max_risk_rounds: int = 1):
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_rounds = max_risk_rounds

    @staticmethod
    def _text(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def _preferred_action(self, context: SchedulerContext) -> SchedulerAction:
        state = context.state
        for analyst in context.selected_analysts:
            action = ANALYST_ACTION_BY_KEY[analyst]
            report = {
                SchedulerAction.MARKET: "market_report",
                SchedulerAction.SENTIMENT: "sentiment_report",
                SchedulerAction.NEWS: "news_report",
                SchedulerAction.FUNDAMENTALS: "fundamentals_report",
            }[action]
            if not self._text(state.get(report)):
                return action

        debate = state.get("investment_debate_state") or {}
        debate_count = int(debate.get("count") or 0)
        if debate_count < 2 * self.max_debate_rounds:
            response = str(debate.get("current_response") or "")
            return SchedulerAction.BEAR if response.startswith("Bull") else SchedulerAction.BULL
        if not self._text(state.get("investment_plan")):
            return SchedulerAction.RESEARCH_MANAGER
        if not self._text(state.get("trader_investment_plan")):
            return SchedulerAction.TRADER

        risk = state.get("risk_debate_state") or {}
        risk_count = int(risk.get("count") or 0)
        if risk_count < 3 * self.max_risk_rounds:
            latest = str(risk.get("latest_speaker") or "")
            if latest.startswith("Aggressive"):
                return SchedulerAction.CONSERVATIVE
            if latest.startswith("Conservative"):
                return SchedulerAction.NEUTRAL
            return SchedulerAction.AGGRESSIVE
        if not self._text(state.get("final_trade_decision")):
            return SchedulerAction.PORTFOLIO_MANAGER
        return SchedulerAction.STOP

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        preferred = self._preferred_action(context)
        if preferred not in context.valid_actions:
            raise ValueError(
                f"static scheduler selected {preferred.value}, not in valid actions "
                f"{[action.value for action in context.valid_actions]}"
            )
        return PolicyDecision(preferred, reason_code="STATIC_PATH")


class CallableSchedulerPolicy:
    """Adapter used by tests and non-HuggingFace policy backends."""

    def __init__(
        self,
        selector: Callable[[SchedulerContext], PolicyDecision | SchedulerAction | str],
        *,
        policy_id: str = "callable-policy",
    ):
        self.selector = selector
        self.policy_id = policy_id

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        result = self.selector(context)
        if isinstance(result, PolicyDecision):
            decision = result
        else:
            decision = PolicyDecision(parse_action(result))
        if decision.action not in context.valid_actions:
            raise ValueError(f"policy selected masked action: {decision.action.value}")
        return decision


class LearnedSchedulerPolicy(CallableSchedulerPolicy):
    """Runtime adapter for a loaded scheduler backend.

    The training package supplies a callable backend after a model and adapter
    are configured. Keeping Transformers out of this module preserves the core
    TradingAgents installation.
    """

    def __init__(
        self,
        selector: Callable[[SchedulerContext], PolicyDecision | SchedulerAction | str],
        *,
        policy_id: str,
    ):
        if not policy_id:
            raise ValueError("learned scheduler policy_id is required")
        super().__init__(selector, policy_id=policy_id)


class ReferenceScoredPolicy:
    """Attach frozen-reference logprobs to decisions made by an active policy."""

    def __init__(self, active: SchedulerPolicy, reference: Any):
        if not hasattr(reference, "action_logprob"):
            raise TypeError("reference policy must implement action_logprob")
        self.active = active
        self.reference = reference
        self.policy_id = active.policy_id

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        decision = self.active.select_action(context)
        metadata = dict(decision.metadata)
        metadata["ref_logprob"] = self.reference.action_logprob(
            context, decision.action
        )
        return PolicyDecision(
            action=decision.action,
            reason_code=decision.reason_code,
            reason=decision.reason,
            logprob=decision.logprob,
            metadata=metadata,
        )
