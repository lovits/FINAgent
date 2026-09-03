"""In-process trajectory recorder attached to the learned Scheduler node."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from .actions import SchedulerAction, node_for_action
from .policy import PolicyDecision, SchedulerContext
from .trajectory import SchedulerTrajectory, TrajectoryStatus, TrajectoryStep
from .trajectory_store import TrajectoryStore


class TrajectoryRecorder:
    def __init__(
        self,
        store_path: str | Path | None = None,
        *,
        stats_provider: Callable[[], dict[str, int]] | None = None,
    ):
        self.store = TrajectoryStore(store_path) if store_path else None
        self.stats_provider = stats_provider
        self.current: SchedulerTrajectory | None = None
        self._last_stats: dict[str, int] = {}
        self._last_timestamp = 0.0

    def begin(
        self,
        *,
        task_id: str,
        ticker: str,
        trade_date: str,
        asset_type: str,
        policy_id: str,
        mode: str = "learned",
        metadata: dict[str, Any] | None = None,
    ) -> SchedulerTrajectory:
        if self.current is not None and self.current.status == "running":
            raise ValueError("cannot begin a trajectory while another is running")
        self.current = SchedulerTrajectory(
            trajectory_id=str(uuid4()),
            task_id=task_id,
            run_id=str(uuid4()),
            mode=mode,
            policy_id=policy_id,
            ticker=ticker,
            trade_date=trade_date,
            asset_type=asset_type,
            metadata=metadata or {},
        )
        self._last_stats = self.stats_provider() if self.stats_provider else {}
        self._last_timestamp = monotonic()
        return self.current

    def record_decision(
        self, context: SchedulerContext, decision: PolicyDecision
    ) -> None:
        if self.current is None or self.current.status != "running":
            raise ValueError("trajectory recorder has not been started")
        if self.current.steps:
            previous = self.current.steps[-1]
            previous.observation_summary = context.serialized_state
            current_stats = self.stats_provider() if self.stats_provider else {}
            previous.tool_calls = max(
                0, current_stats.get("tool_calls", 0) - self._last_stats.get("tool_calls", 0)
            )
            previous.input_tokens = max(
                0,
                current_stats.get("input_tokens", 0)
                - self._last_stats.get("input_tokens", 0),
            )
            previous.output_tokens = max(
                0,
                current_stats.get("output_tokens", 0)
                - self._last_stats.get("output_tokens", 0),
            )
            previous.latency_ms = max(0.0, (monotonic() - self._last_timestamp) * 1000)
            self._last_stats = current_stats
            self._last_timestamp = monotonic()
        agent_node = (
            None
            if decision.action is SchedulerAction.STOP
            else node_for_action(decision.action)
        )
        self.current.add_step(
            TrajectoryStep(
                step_id=len(self.current.steps),
                serialized_state=context.serialized_state,
                valid_actions=[action.value for action in context.valid_actions],
                selected_action=decision.action.value,
                policy_id=self.current.policy_id,
                agent_node=agent_node,
                logprob=decision.logprob,
                metadata=dict(decision.metadata),
            )
        )
        teacher_model = decision.metadata.get("teacher_model")
        if teacher_model:
            self.current.teacher_model = str(teacher_model)
        teacher_prompt_version = decision.metadata.get("teacher_prompt_version")
        if teacher_prompt_version:
            self.current.teacher_prompt_version = str(teacher_prompt_version)

    def finalize(
        self,
        final_state: dict[str, Any],
        *,
        status: TrajectoryStatus,
        failure_reason: str | None = None,
        fallback_reason: str | None = None,
        verifier_version: str | None = None,
    ) -> SchedulerTrajectory:
        if self.current is None:
            raise ValueError("trajectory recorder has not been started")
        if self.current.status != "running":
            raise ValueError("trajectory has already been finalized")
        self.current.status = status
        self.current.investment_plan = str(final_state.get("investment_plan") or "")
        self.current.trader_result = str(final_state.get("trader_investment_plan") or "")
        self.current.final_decision = str(final_state.get("final_trade_decision") or "")
        self.current.failure_reason = failure_reason
        self.current.fallback_reason = fallback_reason
        self.current.verifier_version = verifier_version
        if self.store is not None:
            self.store.append(self.current)
        return self.current
