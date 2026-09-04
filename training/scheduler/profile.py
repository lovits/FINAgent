"""Single supported training profile for scheduler v1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tradingagents.scheduler.prompt import ORCHESTRATION_PROFILE_VERSION
from tradingagents.scheduler.registry import registry_for_analysts


def validate_training_profile(selected_analysts: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(selected_analysts)
    registry_for_analysts(selected)
    if len(selected) < 2:
        raise ValueError(
            f"{ORCHESTRATION_PROFILE_VERSION} requires at least two available analysts"
        )
    return selected


def validate_shallow_runtime(config: Mapping[str, Any]) -> None:
    debate_rounds = int(config.get("max_debate_rounds", 1))
    risk_rounds = int(config.get("max_risk_discuss_rounds", 1))
    if debate_rounds != 1 or risk_rounds != 1:
        raise ValueError(
            f"{ORCHESTRATION_PROFILE_VERSION} requires shallow Static baselines "
            "with max_debate_rounds=1 and max_risk_discuss_rounds=1"
        )
