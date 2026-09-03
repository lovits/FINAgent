"""Load and validate the versioned Strong Teacher context bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .actions import SchedulerAction
from .agent_registry import AGENT_REGISTRY, registry_for_analysts
from .teacher_prompt import TeacherPromptInput


def default_context_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "scheduler" / "teacher" / "v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def load_teacher_context(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(path) if path is not None else default_context_dir()
    context = {
        "agent_catalog": _load_json(root / "agent_catalog.yaml"),
        "state_schema": _load_json(root / "state_schema.yaml"),
        "orchestration_rules": _load_json(root / "orchestration_rules.yaml"),
        "positive_examples": _load_jsonl(root / "positive_examples.jsonl"),
        "failure_examples": _load_jsonl(root / "failure_examples.jsonl"),
    }
    registered = {action.value for action in AGENT_REGISTRY}
    catalog = {item["action"] for item in context["agent_catalog"]["agents"]}
    if catalog != registered:
        raise ValueError("Teacher Agent Catalog does not match the runtime Agent Registry")
    valid_or_stop = registered | {SchedulerAction.STOP.value}
    for example in context["positive_examples"]:
        if example["next_agent"] not in valid_or_stop:
            raise ValueError(f"Teacher positive example uses unknown action: {example}")
    return context


def make_teacher_prompt_input(
    serialized_state: str,
    valid_actions: tuple[SchedulerAction, ...],
    selected_analysts: tuple[str, ...],
    *,
    path: str | Path | None = None,
) -> TeacherPromptInput:
    context = load_teacher_context(path)
    registry = registry_for_analysts(selected_analysts)
    return TeacherPromptInput(
        serialized_state=serialized_state,
        valid_actions=valid_actions,
        agent_specs=tuple(registry.values()),
        hard_rules=tuple(context["orchestration_rules"]["rules"]),
        positive_examples=context["positive_examples"],
        failure_examples=context["failure_examples"],
    )
