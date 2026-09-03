"""Deterministic acceptance checks for generated scheduler trajectories."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .trajectory import SchedulerTrajectory

VERIFIER_VERSION = "v1"
RejectionReason = Literal[
    "invalid",
    "incomplete",
    "loop",
    "budget",
    "quality",
]


@dataclass(frozen=True)
class VerificationLimits:
    max_steps: int = 16
    max_agent_calls: int = 16
    max_tool_calls: int = 40
    max_total_tokens: int = 100_000
    max_same_action: int = 3


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reason: RejectionReason | None
    details: tuple[str, ...]
    verifier_version: str = VERIFIER_VERSION


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def verify_trajectory(
    trajectory: SchedulerTrajectory,
    final_state: Mapping[str, Any],
    *,
    limits: VerificationLimits | None = None,
    quality_ok: bool = True,
) -> VerificationResult:
    """Accept only legal, complete, bounded trajectories with usable outputs."""

    limits = limits or VerificationLimits()
    errors = [step.error for step in trajectory.steps if step.error]
    if errors:
        return VerificationResult(False, "invalid", tuple(str(error) for error in errors))

    if len(trajectory.steps) > limits.max_steps:
        return VerificationResult(False, "budget", ("max_steps exceeded",))
    agent_calls = sum(1 for step in trajectory.steps if step.agent_node)
    tool_calls = sum(step.tool_calls for step in trajectory.steps)
    total_tokens = sum(step.input_tokens + step.output_tokens for step in trajectory.steps)
    if (
        agent_calls > limits.max_agent_calls
        or tool_calls > limits.max_tool_calls
        or total_tokens > limits.max_total_tokens
    ):
        return VerificationResult(False, "budget", ("trajectory cost limit exceeded",))

    counts = Counter(step.selected_action for step in trajectory.steps)
    if any(count > limits.max_same_action for count in counts.values()):
        return VerificationResult(False, "loop", ("same action repeated too often",))
    if any(step.no_progress for step in trajectory.steps):
        return VerificationResult(False, "loop", ("trajectory contains no-progress step",))

    missing = [
        field
        for field in ("investment_plan", "trader_investment_plan", "final_trade_decision")
        if not _text(final_state.get(field))
    ]
    if missing:
        return VerificationResult(False, "incomplete", tuple(f"missing {field}" for field in missing))
    if not quality_ok:
        return VerificationResult(False, "quality", ("quality gate failed",))
    return VerificationResult(True, None, ())
