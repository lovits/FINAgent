"""Translate scheduler decisions and Expert observations into one trajectory."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .actions import SchedulerAction, node_for_action
from .contracts import PolicyDecision, SchedulerContext
from .prompt import business_state
from .trajectory import (
    ExecutionCost,
    ExecutionStatus,
    NodeExecution,
    SchedulerStep,
    SchedulerTrajectory,
)


class TrajectoryRecorder:
    def __init__(self, trajectory: SchedulerTrajectory):
        self.trajectory = trajectory
        self._pending: SchedulerStep | None = None

    @property
    def pending_agent_node(self) -> str | None:
        return None if self._pending is None else self._pending.agent_node

    @property
    def has_pending_observation(self) -> bool:
        return self._pending is not None

    def record_decision(
        self, context: SchedulerContext, decision: PolicyDecision
    ) -> None:
        if self._pending is not None:
            raise ValueError("previous scheduler decision has no Expert observation")
        action = decision.action
        step = SchedulerStep(
            step_id=len(self.trajectory.steps),
            state_before=deepcopy(business_state(context.state)),
            serialized_state=context.serialized_state,
            valid_actions=[value.value for value in context.valid_actions],
            selected_action=action.value,
            agent_node=None if action is SchedulerAction.STOP else node_for_action(action),
            decision_attempts=decision.decision_attempts,
            correction_succeeded=decision.correction_succeeded,
            old_logprob=decision.logprob,
            ref_logprob=self._optional_float(decision.metadata.get("ref_logprob")),
            metadata=dict(decision.metadata),
        )
        if action is SchedulerAction.STOP:
            step.state_after = deepcopy(step.state_before)
            self.trajectory.add_step(step)
        else:
            self._pending = step

    def record_observation(
        self,
        state_after: dict[str, Any],
        *,
        observation_ref: str,
        cost: ExecutionCost | None = None,
        error: str | None = None,
    ) -> None:
        if self._pending is None:
            raise ValueError("Expert observation has no pending scheduler decision")
        self._pending.state_after = deepcopy(business_state(state_after))
        self._pending.observation_ref = observation_ref
        self._pending.cost = cost or ExecutionCost(agent_calls=1)
        self._pending.error = error
        self.trajectory.add_step(self._pending)
        self._pending = None

    def record_node_execution(self, execution: NodeExecution) -> None:
        self.trajectory.add_node_execution(execution)

    def finalize(
        self,
        final_state: dict[str, Any],
        *,
        status: ExecutionStatus = "completed",
        failure_reason: str | None = None,
    ) -> SchedulerTrajectory:
        if self._pending is not None:
            raise ValueError("cannot finalize with a pending Expert observation")
        self.trajectory.execution_status = status
        self.trajectory.failure_reason = failure_reason
        self.trajectory.final_outputs = {
            "investment_plan": final_state.get("investment_plan") or "",
            "trader_investment_plan": final_state.get("trader_investment_plan") or "",
            "final_trade_decision": final_state.get("final_trade_decision") or "",
        }
        total = ExecutionCost()
        for step in self.trajectory.steps:
            total += step.cost
        self.trajectory.cost_total = total
        return self.trajectory

    @staticmethod
    def _optional_float(value: object) -> float | None:
        return None if value is None else float(value)
