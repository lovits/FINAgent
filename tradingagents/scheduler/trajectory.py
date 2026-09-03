"""Versioned scheduler trajectory records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .actions import parse_action

TRAJECTORY_SCHEMA_VERSION = "v1"
TrajectoryStatus = Literal["running", "accepted", "rejected", "failed", "fallback"]


@dataclass
class TrajectoryStep:
    step_id: int
    serialized_state: str
    valid_actions: list[str]
    selected_action: str
    policy_id: str
    agent_node: str | None = None
    logprob: float | None = None
    observation_summary: str = ""
    observation_ref: str | None = None
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    no_progress: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_id < 0:
            raise ValueError("step_id cannot be negative")
        selected = parse_action(self.selected_action).value
        valid = [parse_action(action).value for action in self.valid_actions]
        if selected not in valid:
            raise ValueError(f"selected action {selected} is not valid at step {self.step_id}")
        self.selected_action = selected
        self.valid_actions = valid


@dataclass
class SchedulerTrajectory:
    trajectory_id: str
    task_id: str
    run_id: str
    mode: Literal["static", "learned"]
    policy_id: str
    ticker: str
    trade_date: str
    asset_type: str = "stock"
    data_snapshot_id: str | None = None
    action_schema_version: str = "v1"
    state_schema_version: str = "v1"
    teacher_model: str | None = None
    teacher_prompt_version: str | None = None
    verifier_version: str | None = None
    status: TrajectoryStatus = "running"
    steps: list[TrajectoryStep] = field(default_factory=list)
    investment_plan: str = ""
    trader_result: str = ""
    final_decision: str = ""
    reward: dict[str, float] = field(default_factory=dict)
    failure_reason: str | None = None
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    def add_step(self, step: TrajectoryStep) -> None:
        if self.status != "running":
            raise ValueError("cannot append to a terminal trajectory")
        expected = len(self.steps)
        if step.step_id != expected:
            raise ValueError(f"expected step_id {expected}, got {step.step_id}")
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SchedulerTrajectory:
        payload = dict(value)
        version = payload.get("schema_version")
        if version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema: {version!r}")
        payload["steps"] = [TrajectoryStep(**step) for step in payload.get("steps", [])]
        return cls(**payload)
