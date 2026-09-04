"""Create deterministic, balanced task plans for Scheduler trajectory collection."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tradingagents.scheduler.store import write_json_atomic

from .generate import load_tasks

COLLECTION_PLAN_VERSION = "scheduler-collection-plan-v1"


def balanced_candidates(
    tasks: Iterable[Mapping[str, Any]], *, source_split: str
) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for task in tasks:
        if task.get("split") != source_split:
            continue
        key = str(task.get("seed_family"))
        buckets[key].append(dict(task))
    ordered: list[dict[str, Any]] = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                ordered.append(buckets[key].popleft())
    return ordered


def build_collection_tasks(
    tasks: Iterable[Mapping[str, Any]],
    *,
    paired_count: int,
    teacher_only_count: int,
    source_split: str = "reserve",
    target_split: str = "train",
    offset: int = 0,
) -> list[dict[str, Any]]:
    if min(paired_count, teacher_only_count, offset) < 0:
        raise ValueError("collection counts and offset must be non-negative")
    required = paired_count + teacher_only_count
    candidates = balanced_candidates(tasks, source_split=source_split)
    selected = candidates[offset : offset + required]
    if len(selected) != required:
        raise ValueError(f"only {len(selected)} tasks available; required {required}")
    values = []
    for index, task in enumerate(selected):
        task["source_split"] = task["split"]
        task["split"] = target_split
        task["collection_role"] = "paired" if index < paired_count else "teacher_only"
        values.append(task)
    return values


def write_collection_tasks(
    tasks: Sequence[Mapping[str, Any]], output_path: str | Path
) -> dict[str, Any]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(dict(task), ensure_ascii=False, sort_keys=True) + "\n")
    role_counts = Counter(str(task["collection_role"]) for task in tasks)
    manifest = {
        "schema_version": COLLECTION_PLAN_VERSION,
        "task_count": len(tasks),
        "planned_trajectory_count": role_counts["paired"] * 2 + role_counts["teacher_only"],
        "role_counts": dict(role_counts),
        "source_split_counts": dict(Counter(str(task["source_split"]) for task in tasks)),
        "seed_family_counts": dict(Counter(str(task["seed_family"]) for task in tasks)),
        "analyst_count_counts": dict(
            sorted(Counter(len(task["selected_analysts"]) for task in tasks).items())
        ),
    }
    write_json_atomic(destination.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-count", type=int, required=True)
    parser.add_argument("--teacher-only-count", type=int, required=True)
    parser.add_argument("--source-split", default="reserve")
    parser.add_argument("--target-split", default="train")
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    tasks = build_collection_tasks(
        load_tasks(args.pool, split=None),
        paired_count=args.paired_count,
        teacher_only_count=args.teacher_only_count,
        source_split=args.source_split,
        target_split=args.target_split,
        offset=args.offset,
    )
    manifest = write_collection_tasks(tasks, args.output)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
