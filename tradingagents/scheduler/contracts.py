"""Runtime policy contracts shared by Teacher and local scheduler backends."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .actions import SchedulerAction, parse_action


@dataclass(frozen=True)
class SchedulerContext:
    task_id: str
    state: Mapping[str, Any]
    serialized_state: str
    valid_actions: tuple[SchedulerAction, ...]
    selected_analysts: tuple[str, ...]
    history: tuple[SchedulerAction, ...] = ()
    step: int = 0
    max_steps: int = 16
    no_progress_count: int = 0

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("scheduler context requires a task_id")
        if not self.valid_actions:
            raise ValueError("scheduler context requires at least one valid action")
        if self.step < 0 or self.max_steps <= 0 or self.step > self.max_steps:
            raise ValueError("invalid scheduler step budget")
        if self.no_progress_count < 0:
            raise ValueError("no_progress_count cannot be negative")
        object.__setattr__(
            self,
            "valid_actions",
            tuple(parse_action(action) for action in self.valid_actions),
        )
        object.__setattr__(
            self,
            "history",
            tuple(parse_action(action) for action in self.history),
        )


@dataclass(frozen=True)
class PolicyDecision:
    action: SchedulerAction
    policy_id: str
    logprob: float | None = None
    decision_attempts: int = 1
    correction_succeeded: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", parse_action(self.action))
        if not self.policy_id:
            raise ValueError("policy decision requires a policy_id")
        if self.decision_attempts not in (1, 2):
            raise ValueError("decision_attempts must be one or two")
        if self.correction_succeeded and self.decision_attempts != 2:
            raise ValueError("correction_succeeded requires two decision attempts")
        if self.logprob is not None and not math.isfinite(self.logprob):
            raise ValueError("policy logprob must be finite")


@runtime_checkable
class SchedulerPolicy(Protocol):
    policy_id: str

    def select_action(self, context: SchedulerContext) -> PolicyDecision: ...


@runtime_checkable
class ActionLogprobPolicy(SchedulerPolicy, Protocol):
    def action_logprobs(
        self, context: SchedulerContext
    ) -> Mapping[SchedulerAction, float]: ...
