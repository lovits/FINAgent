"""Canonical, versioned execution records for scheduler data and training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .actions import SchedulerAction, node_for_action, parse_action

TRAJECTORY_SCHEMA_VERSION = "scheduler-trajectory-v1"
ExecutionStatus = Literal[
    "running",
    "completed",
    "failed",
    "fallback",
    "budget_exhausted",
    "context_overflow",
]
AuditStatus = Literal["pending", "accepted", "rejected", "warning"]
TrajectoryMode = Literal["static", "teacher", "learned"]


@dataclass(frozen=True)
class ExecutionCost:
    agent_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.agent_calls,
            self.tool_calls,
            self.input_tokens,
            self.output_tokens,
            self.latency_ms,
        )
        if any(value < 0 for value in values):
            raise ValueError("execution costs cannot be negative")
        if not math.isfinite(self.latency_ms):
            raise ValueError("latency_ms must be finite")

    def __add__(self, other: ExecutionCost) -> ExecutionCost:
        return ExecutionCost(
            agent_calls=self.agent_calls + other.agent_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
        )


@dataclass
class NodeExecution:
    node_step_id: int
    node_name: str
    node_type: Literal["scheduler", "expert", "tool", "message_cleanup"]
    state_before: dict[str, Any]
    state_update: dict[str, Any]
    state_after: dict[str, Any]
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    cost: ExecutionCost = field(default_factory=ExecutionCost)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.node_step_id < 0 or not self.node_name:
            raise ValueError("node execution requires a non-negative id and node name")


@dataclass
class SchedulerStep:
    step_id: int
    state_before: dict[str, Any]
    serialized_state: str
    valid_actions: list[str]
    selected_action: str
    agent_node: str | None
    state_after: dict[str, Any] = field(default_factory=dict)
    decision_attempts: int = 1
    correction_succeeded: bool = False
    observation_ref: str | None = None
    old_logprob: float | None = None
    ref_logprob: float | None = None
    cost: ExecutionCost = field(default_factory=ExecutionCost)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("scheduler step_id cannot be negative")
        action = parse_action(self.selected_action)
        self.selected_action = action.value
        self.valid_actions = [parse_action(value).value for value in self.valid_actions]
        if self.selected_action not in self.valid_actions:
            raise ValueError("selected action is not present in valid_actions")
        if action is SchedulerAction.STOP:
            if self.agent_node is not None:
                raise ValueError("STOP cannot reference an Expert Agent node")
        elif self.agent_node != node_for_action(action):
            raise ValueError("agent_node does not match selected_action")
        if self.decision_attempts not in (1, 2):
            raise ValueError("decision_attempts must be one or two")
        if self.correction_succeeded and self.decision_attempts != 2:
            raise ValueError("corrected decisions require two attempts")
        for value in (self.old_logprob, self.ref_logprob):
            if value is not None and not math.isfinite(value):
                raise ValueError("scheduler logprobs must be finite")


@dataclass
class SchedulerTrajectory:
    trajectory_id: str
    run_id: str
    task_id: str
    mode: TrajectoryMode
    policy_id: str
    ticker: str
    trade_date: str
    asset_type: str = "stock"
    data_snapshot_id: str | None = None
    execution_status: ExecutionStatus = "running"
    audit_status: AuditStatus = "pending"
    steps: list[SchedulerStep] = field(default_factory=list)
    node_executions: list[NodeExecution] = field(default_factory=list)
    final_outputs: dict[str, Any] = field(default_factory=dict)
    cost_total: ExecutionCost = field(default_factory=ExecutionCost)
    reward: dict[str, float] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all((self.trajectory_id, self.run_id, self.task_id, self.policy_id)):
            raise ValueError("trajectory identifiers and policy_id are required")
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema: {self.schema_version}")

    def add_step(self, step: SchedulerStep) -> None:
        if self.execution_status != "running":
            raise ValueError("cannot add steps to a terminal trajectory")
        if step.step_id != len(self.steps):
            raise ValueError(f"expected scheduler step {len(self.steps)}, got {step.step_id}")
        self.steps.append(step)

    def add_node_execution(self, execution: NodeExecution) -> None:
        if execution.node_step_id != len(self.node_executions):
            raise ValueError(
                f"expected node step {len(self.node_executions)}, got {execution.node_step_id}"
            )
        self.node_executions.append(execution)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SchedulerTrajectory:
        payload = dict(value)
        payload["steps"] = [
            SchedulerStep(**{**step, "cost": ExecutionCost(**step.get("cost", {}))})
            for step in payload.get("steps", [])
        ]
        payload["node_executions"] = [
            NodeExecution(
                **{**item, "cost": ExecutionCost(**item.get("cost", {}))}
            )
            for item in payload.get("node_executions", [])
        ]
        payload["cost_total"] = ExecutionCost(**payload.get("cost_total", {}))
        return cls(**payload)
