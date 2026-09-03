"""Run Static or policy-driven TradingAgents graphs and capture full trajectories."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import uuid4

from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.scheduler.action_mask import compute_action_mask
from tradingagents.scheduler.actions import ACTION_BY_NODE, SchedulerAction
from tradingagents.scheduler.contracts import PolicyDecision, SchedulerContext, SchedulerPolicy
from tradingagents.scheduler.prompt import build_scheduler_input, business_state
from tradingagents.scheduler.recorder import TrajectoryRecorder
from tradingagents.scheduler.trajectory import (
    ExecutionCost,
    NodeExecution,
    SchedulerTrajectory,
)


@dataclass(frozen=True)
class EnvironmentRunResult:
    trajectory: SchedulerTrajectory
    final_state: dict[str, Any]


def _json_safe(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _node_type(node_name: str) -> str:
    if node_name == "Scheduler":
        return "scheduler"
    if node_name.startswith("tools_"):
        return "tool"
    if node_name.startswith("Msg Clear"):
        return "message_cleanup"
    return "expert"


_CLEAR_BY_ANALYST = {
    spec.agent_node: spec.clear_node for spec in ANALYST_NODE_SPECS.values()
}


class TradingAgentsSchedulerEnvironment:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        selected_analysts: tuple[str, ...] = (
            "market",
            "social",
            "news",
            "fundamentals",
        ),
        graph_factory=TradingAgentsGraph,
    ):
        self.config = dict(config)
        self.selected_analysts = selected_analysts
        self.graph_factory = graph_factory

    def run(
        self,
        task: dict[str, Any],
        *,
        mode: str,
        run_id: str,
        policy: SchedulerPolicy | None = None,
        past_context: str = "",
        trajectory_id: str | None = None,
    ) -> EnvironmentRunResult:
        if mode not in {"static", "teacher", "learned"}:
            raise ValueError(f"unsupported trajectory mode: {mode!r}")
        if mode != "static" and policy is None:
            raise ValueError(f"{mode} trajectory generation requires a policy")

        identifier = trajectory_id or str(uuid4())
        policy_id = "static-langgraph-v1" if policy is None else policy.policy_id
        trajectory = SchedulerTrajectory(
            trajectory_id=identifier,
            run_id=run_id,
            task_id=str(task["task_id"]),
            mode=mode,
            policy_id=policy_id,
            ticker=str(task["ticker"]),
            trade_date=str(task["trade_date"]),
            asset_type=str(task.get("asset_type", "stock")),
            data_snapshot_id=task.get("data_snapshot_id"),
            provenance={"task_dataset_version": task.get("dataset_version")},
        )
        recorder = TrajectoryRecorder(trajectory)
        runtime_config = {
            **self.config,
            "orchestration_mode": mode,
            "memory_log_path": None,
            "checkpoint_enabled": False,
        }
        graph = self.graph_factory(
            selected_analysts=self.selected_analysts,
            config=runtime_config,
            scheduler_policy=policy,
            scheduler_on_decision=recorder.record_decision,
        )
        initial_state = graph.create_initial_state(
            trajectory.ticker,
            trajectory.trade_date,
            asset_type=trajectory.asset_type,
            past_context=past_context,
            scheduler_task_id=trajectory.task_id,
        )

        try:
            final_state = self._capture(graph, initial_state, recorder, mode)
            if mode == "static":
                self._record_static_stop(final_state, recorder)
            status = "completed" if final_state.get("final_trade_decision") else "failed"
            failure = None if status == "completed" else "final_decision_missing"
        except Exception as exc:
            final_state = locals().get("final_state", initial_state)
            if recorder.has_pending_observation:
                recorder.record_observation(
                    final_state,
                    observation_ref="execution-error",
                    cost=ExecutionCost(agent_calls=1),
                    error=f"{type(exc).__name__}: {exc}",
                )
            status = "failed"
            failure = f"{type(exc).__name__}: {exc}"
        return EnvironmentRunResult(
            recorder.finalize(final_state, status=status, failure_reason=failure),
            final_state,
        )

    def _capture(
        self,
        graph: TradingAgentsGraph,
        initial_state: dict[str, Any],
        recorder: TrajectoryRecorder,
        mode: str,
    ) -> dict[str, Any]:
        current_state = deepcopy(initial_state)
        pending_updates: list[tuple[str, object, dict[str, Any], float]] = []
        observation_ref: str | None = None
        pending_tool_calls = 0
        decision_started: float | None = None
        arguments = graph.propagator.get_graph_args()
        arguments["stream_mode"] = ["updates", "values"]

        for stream_mode, payload in graph.graph.stream(initial_state, **arguments):
            if stream_mode == "updates":
                for node_name, update in payload.items():
                    if mode == "static" and node_name in ACTION_BY_NODE:
                        self._start_static_decision(current_state, node_name, recorder)
                    pending_updates.append(
                        (node_name, update, deepcopy(business_state(current_state)), monotonic())
                    )
                continue
            if stream_mode != "values":
                continue

            next_state = dict(payload)
            for node_name, update, state_before, started_at in pending_updates:
                node_id = len(recorder.trajectory.node_executions)
                node_kind = _node_type(node_name)
                node_cost = ExecutionCost(
                    tool_calls=1 if node_kind == "tool" else 0,
                    latency_ms=max(0.0, (monotonic() - started_at) * 1000),
                )
                recorder.record_node_execution(
                    NodeExecution(
                        node_step_id=node_id,
                        node_name=node_name,
                        node_type=node_kind,
                        state_before=state_before,
                        state_update=_json_safe(update),
                        state_after=deepcopy(business_state(next_state)),
                        cost=node_cost,
                    )
                )
                if recorder.has_pending_observation and decision_started is None:
                    decision_started = started_at
                if node_kind == "tool" and recorder.has_pending_observation:
                    pending_tool_calls += 1
                if node_name == recorder.pending_agent_node:
                    observation_ref = f"node-{node_id}"
                if self._completes_pending(node_name, recorder.pending_agent_node):
                    recorder.record_observation(
                        next_state,
                        observation_ref=observation_ref or f"node-{node_id}",
                        cost=ExecutionCost(
                            agent_calls=1,
                            tool_calls=pending_tool_calls,
                            latency_ms=max(
                                0.0,
                                (monotonic() - (decision_started or started_at)) * 1000,
                            ),
                        ),
                    )
                    observation_ref = None
                    pending_tool_calls = 0
                    decision_started = None
            pending_updates.clear()
            current_state = next_state
        return current_state

    @staticmethod
    def _completes_pending(node_name: str, pending_agent: str | None) -> bool:
        if pending_agent is None:
            return False
        clear_node = _CLEAR_BY_ANALYST.get(pending_agent)
        return node_name == (clear_node or pending_agent)

    def _start_static_decision(
        self,
        state: dict[str, Any],
        node_name: str,
        recorder: TrajectoryRecorder,
    ) -> None:
        if recorder.has_pending_observation:
            return
        action = ACTION_BY_NODE[node_name]
        step = len(recorder.trajectory.steps)
        mask = compute_action_mask(
            state,
            self.selected_analysts,
            step=step,
            max_steps=int(self.config.get("scheduler_max_steps", 16)),
            max_debate_rounds=int(self.config.get("max_debate_rounds", 1)),
            max_risk_rounds=int(self.config.get("max_risk_discuss_rounds", 1)),
        )
        if action not in mask.valid_actions:
            raise ValueError(f"static node {node_name} is invalid at scheduler step {step}")
        history = tuple(item.selected_action for item in recorder.trajectory.steps)
        serialized = build_scheduler_input(
            task_id=recorder.trajectory.task_id,
            state=state,
            valid_actions=mask.valid_actions,
            selected_analysts=self.selected_analysts,
            history=history,
            step=step,
            max_steps=int(self.config.get("scheduler_max_steps", 16)),
        )
        context = SchedulerContext(
            recorder.trajectory.task_id,
            state,
            serialized,
            mask.valid_actions,
            self.selected_analysts,
            history=history,
            step=step,
            max_steps=int(self.config.get("scheduler_max_steps", 16)),
        )
        recorder.record_decision(
            context,
            PolicyDecision(action, policy_id=recorder.trajectory.policy_id),
        )

    def _record_static_stop(
        self, final_state: dict[str, Any], recorder: TrajectoryRecorder
    ) -> None:
        step = len(recorder.trajectory.steps)
        mask = compute_action_mask(
            final_state,
            self.selected_analysts,
            step=step,
            max_steps=int(self.config.get("scheduler_max_steps", 16)),
        )
        if mask.valid_actions != (SchedulerAction.STOP,):
            raise ValueError("static graph completed without a valid STOP state")
        history = tuple(item.selected_action for item in recorder.trajectory.steps)
        serialized = build_scheduler_input(
            task_id=recorder.trajectory.task_id,
            state=final_state,
            valid_actions=mask.valid_actions,
            selected_analysts=self.selected_analysts,
            history=history,
            step=step,
            max_steps=int(self.config.get("scheduler_max_steps", 16)),
        )
        context = SchedulerContext(
            recorder.trajectory.task_id,
            final_state,
            serialized,
            mask.valid_actions,
            self.selected_analysts,
            history=history,
            step=step,
            max_steps=int(self.config.get("scheduler_max_steps", 16)),
        )
        recorder.record_decision(
            context,
            PolicyDecision(SchedulerAction.STOP, policy_id=recorder.trajectory.policy_id),
        )
