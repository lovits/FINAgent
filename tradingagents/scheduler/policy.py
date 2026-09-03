"""Small policy adapters used by tests and non-Hugging Face integrations."""

from __future__ import annotations

from collections.abc import Callable

from .actions import SchedulerAction, parse_action
from .contracts import ActionLogprobPolicy, PolicyDecision, SchedulerContext, SchedulerPolicy


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


class ReferenceScoredPolicy:
    """Attach frozen-reference logprob to decisions sampled by the active policy."""

    def __init__(
        self,
        active: SchedulerPolicy,
        reference: ActionLogprobPolicy,
    ):
        self.active = active
        self.reference = reference
        self.policy_id = active.policy_id

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        decision = self.active.select_action(context)
        reference_logprobs = self.reference.action_logprobs(context)
        if decision.action not in reference_logprobs:
            raise ValueError("reference policy did not score the selected action")
        return PolicyDecision(
            decision.action,
            policy_id=decision.policy_id,
            logprob=decision.logprob,
            decision_attempts=decision.decision_attempts,
            correction_succeeded=decision.correction_succeeded,
            metadata={
                **decision.metadata,
                "ref_logprob": reference_logprobs[decision.action],
            },
        )
