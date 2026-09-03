"""Small policy adapters used by tests and non-Hugging Face integrations."""

from __future__ import annotations

from collections.abc import Callable

from .actions import SchedulerAction, parse_action
from .contracts import PolicyDecision, SchedulerContext


class CallableSchedulerPolicy:
    def __init__(
        self,
        selector: Callable[[SchedulerContext], PolicyDecision | SchedulerAction | str],
        *,
        policy_id: str = "callable-scheduler",
    ):
        if not policy_id:
            raise ValueError("policy_id is required")
        self.selector = selector
        self.policy_id = policy_id

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        result = self.selector(context)
        if isinstance(result, PolicyDecision):
            decision = result
        else:
            decision = PolicyDecision(parse_action(result), policy_id=self.policy_id)
        if decision.action not in context.valid_actions:
            raise ValueError(f"policy selected masked action: {decision.action.value}")
        return decision
