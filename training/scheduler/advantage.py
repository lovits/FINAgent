"""Group-relative trajectory advantages without a learned critic."""

from __future__ import annotations

import math
from collections.abc import Sequence


def group_relative_advantages(
    rewards: Sequence[float], *, epsilon: float = 1e-6
) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("group-relative advantages require at least two rewards")
    if not all(math.isfinite(value) for value in rewards):
        raise ValueError("rewards must be finite")
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    standard_deviation = math.sqrt(variance)
    if standard_deviation < epsilon:
        return [0.0] * len(rewards)
    return [(value - mean) / standard_deviation for value in rewards]
