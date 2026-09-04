"""Single supported training profile for scheduler v1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tradingagents.scheduler.prompt import ORCHESTRATION_PROFILE_VERSION
from tradingagents.scheduler.registry import registry_for_analysts

SHALLOW_RESEARCH_DEPTH = "shallow"
CHINESE_OUTPUT_LANGUAGE = "Chinese"


def validate_training_profile(selected_analysts: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(selected_analysts)
    registry_for_analysts(selected)
    return selected


def resolve_task_runtime(
    task: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...]]:
    raw_analysts = task.get("selected_analysts")
    if not isinstance(raw_analysts, (list, tuple)) or not all(
        isinstance(value, str) for value in raw_analysts
    ):
        raise ValueError("scheduler task requires selected_analysts as a list")
    selected_analysts = validate_training_profile(raw_analysts)

    research_depth = task.get("research_depth")
    if research_depth != SHALLOW_RESEARCH_DEPTH:
        raise ValueError("scheduler profile only supports research_depth=shallow")
    output_language = task.get("output_language")
    if output_language != CHINESE_OUTPUT_LANGUAGE:
        raise ValueError("scheduler profile only supports output_language=Chinese")

    runtime_config = {
        **config,
        "output_language": output_language,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    }
    validate_shallow_runtime(runtime_config)
    return runtime_config, selected_analysts


def validate_shallow_runtime(config: Mapping[str, Any]) -> None:
    debate_rounds = int(config.get("max_debate_rounds", 1))
    risk_rounds = int(config.get("max_risk_discuss_rounds", 1))
    if debate_rounds != 1 or risk_rounds != 1:
        raise ValueError(
            f"{ORCHESTRATION_PROFILE_VERSION} requires shallow Static baselines "
            "with max_debate_rounds=1 and max_risk_discuss_rounds=1"
        )
