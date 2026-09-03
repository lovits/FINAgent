"""Convert accepted scheduler trajectories into masked-action SFT examples."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import quantiles
from typing import Any, Literal

from tradingagents.scheduler.store import write_json_atomic
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .pair import PairComparison

SFT_SCHEMA_VERSION = "scheduler-sft-v1"
SFTSource = Literal["static", "teacher_verified"]


@dataclass(frozen=True)
class SFTExample:
    sample_id: str
    task_id: str
    trajectory_id: str
    step_id: int
    source: SFTSource
    input_text: str
    valid_actions: tuple[str, ...]
    target_action: str
    input_token_count: int
    schema_version: str = SFT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sft_examples(
    trajectories: Iterable[SchedulerTrajectory],
    *,
    source: SFTSource,
    token_counter: Callable[[str], int],
    max_tokens: int = 32768,
    verified_teacher_ids: set[str] | None = None,
) -> tuple[list[SFTExample], int]:
    examples = []
    seen: set[str] = set()
    overflow = 0
    for trajectory in trajectories:
        if trajectory.audit_status not in {"accepted", "warning"}:
            continue
        if source == "teacher_verified" and (
            verified_teacher_ids is None
            or trajectory.trajectory_id not in verified_teacher_ids
        ):
            continue
        for step in trajectory.steps:
            token_count = token_counter(step.serialized_state)
            if token_count > max_tokens:
                overflow += 1
                continue
            dedupe_key = _dedupe_key(step.serialized_state, step.selected_action)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            examples.append(
                SFTExample(
                    sample_id=_sample_id(trajectory.trajectory_id, step.step_id),
                    task_id=trajectory.task_id,
                    trajectory_id=trajectory.trajectory_id,
                    step_id=step.step_id,
                    source=source,
                    input_text=step.serialized_state,
                    valid_actions=tuple(step.valid_actions),
                    target_action=step.selected_action,
                    input_token_count=token_count,
                )
            )
    return examples, overflow


def verified_teacher_ids(comparisons: Iterable[PairComparison]) -> set[str]:
    return {
        comparison.teacher_trajectory_id
        for comparison in comparisons
        if comparison.pair_status == "teacher_verified"
    }


def write_sft_dataset(
    examples: Sequence[SFTExample],
    output_path: str | Path,
    *,
    overflow_count: int = 0,
    source_run_ids: Sequence[str] = (),
) -> dict[str, Any]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    manifest = _manifest(
        examples,
        overflow_count=overflow_count,
        source_run_ids=source_run_ids,
    )
    write_json_atomic(destination.with_suffix(".manifest.json"), manifest)
    return manifest


def _manifest(
    examples: Sequence[SFTExample],
    *,
    overflow_count: int,
    source_run_ids: Sequence[str],
) -> dict[str, Any]:
    lengths = sorted(example.input_token_count for example in examples)
    return {
        "schema_version": "scheduler-sft-manifest-v1",
        "sample_count": len(examples),
        "task_count": len({example.task_id for example in examples}),
        "source_counts": dict(Counter(example.source for example in examples)),
        "action_counts": dict(Counter(example.target_action for example in examples)),
        "length_percentiles": _percentiles(lengths),
        "rejected_context_overflow": overflow_count,
        "source_run_ids": list(source_run_ids),
    }


def _percentiles(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"p50": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    if len(values) == 1:
        return dict.fromkeys(("p50", "p90", "p95", "p99", "max"), values[0])
    cuts = quantiles(values, n=100, method="inclusive")
    return {
        "p50": round(cuts[49]),
        "p90": round(cuts[89]),
        "p95": round(cuts[94]),
        "p99": round(cuts[98]),
        "max": values[-1],
    }


def _sample_id(trajectory_id: str, step_id: int) -> str:
    digest = hashlib.sha256(f"{trajectory_id}:{step_id}".encode()).hexdigest()[:16]
    return f"sft-{digest}"


def _dedupe_key(input_text: str, target_action: str) -> str:
    return hashlib.sha256(f"{input_text}\0{target_action}".encode()).hexdigest()


def load_sft_examples(path: str | Path) -> list[SFTExample]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value: Mapping[str, Any] = json.loads(line)
                rows.append(
                    SFTExample(
                        **{
                            **value,
                            "valid_actions": tuple(value["valid_actions"]),
                        }
                    )
                )
    return rows
