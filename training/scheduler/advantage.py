"""Group-relative trajectory advantage without a learned critic."""

from __future__ import annotations

import math
from collections.abc import Sequence


def group_relative_advantages(
    rewards: Sequence[float], *, epsilon: float = 1e-8
) -> list[float]:
    if len(rewards) < 2:
        raise ValueError("group-relative advantage requires at least two rewards")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < epsilon:
        return [0.0 for _ in rewards]
    return [(reward - mean) / (std + epsilon) for reward in rewards]
