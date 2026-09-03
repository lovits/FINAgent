"""Offline logical replay utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .actions import SchedulerAction, parse_action
from .trajectory import SchedulerTrajectory


@dataclass(frozen=True)
class ReplayDecision:
    step_id: int
    serialized_state: str
    valid_actions: tuple[SchedulerAction, ...]
    recorded_action: SchedulerAction


def replay_decisions(trajectory: SchedulerTrajectory) -> Iterator[ReplayDecision]:
    """Yield recorded state/action pairs without invoking experts or tools."""

    for step in trajectory.steps:
        yield ReplayDecision(
            step_id=step.step_id,
            serialized_state=step.serialized_state,
            valid_actions=tuple(parse_action(action) for action in step.valid_actions),
            recorded_action=parse_action(step.selected_action),
        )
