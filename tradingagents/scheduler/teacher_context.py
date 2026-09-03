"""Load the small, versioned Teacher few-shot context bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .actions import SchedulerAction, parse_action


def default_teacher_context_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "scheduler" / "teacher" / "v1"


def load_teacher_examples(
    context_dir: str | Path | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    root = Path(context_dir) if context_dir is not None else default_teacher_context_dir()
    positive = _load_jsonl(root / "positive_examples.jsonl")
    failures = _load_jsonl(root / "failure_examples.jsonl")
    allowed = {action.value for action in SchedulerAction}
    for example in positive:
        output = example.get("output", {})
        action = parse_action(output.get("action"))
        if action.value not in allowed:
            raise ValueError(f"unknown action in Teacher example: {action.value}")
    return positive, failures


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            values.append(value)
    if not values:
        raise ValueError(f"Teacher example file is empty: {path}")
    return tuple(values)
