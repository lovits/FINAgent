"""Create task-aligned Static/Teacher comparison records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .pair import PairComparison, pair_trajectories


def pair_datasets(
    static: Iterable[SchedulerTrajectory],
    teacher: Iterable[SchedulerTrajectory],
) -> list[PairComparison]:
    static_by_key = _by_task_snapshot(static, expected_mode="static")
    teacher_by_key = _by_task_snapshot(teacher, expected_mode="teacher")
    comparisons = []
    for key in sorted(static_by_key.keys() & teacher_by_key.keys()):
        comparisons.append(pair_trajectories(static_by_key[key], teacher_by_key[key]))
    return comparisons


def _by_task_snapshot(
    trajectories: Iterable[SchedulerTrajectory], *, expected_mode: str
) -> dict[tuple[str, str | None], SchedulerTrajectory]:
    values = {}
    for trajectory in trajectories:
        if trajectory.mode != expected_mode:
            raise ValueError(
                f"expected {expected_mode} trajectory, got {trajectory.mode}"
            )
        key = (trajectory.task_id, trajectory.data_snapshot_id)
        if key in values:
            raise ValueError(f"duplicate {expected_mode} trajectory for {key}")
        values[key] = trajectory
    return values


def write_comparisons(
    comparisons: Iterable[PairComparison], path: str | Path
) -> dict[str, object]:
    values = list(comparisons)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for comparison in values:
            handle.write(json.dumps(comparison.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    manifest = {
        "pairing_version": "scheduler-pair-v1",
        "comparison_count": len(values),
        "status_counts": dict(Counter(value.pair_status for value in values)),
    }
    write_json_atomic(destination.with_suffix(".manifest.json"), manifest)
    return manifest


def load_comparisons(path: str | Path) -> list[PairComparison]:
    values = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["rejection_reasons"] = tuple(row.get("rejection_reasons", ()))
            values.append(PairComparison(**row))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    comparisons = pair_datasets(
        TrajectoryStore(args.static).load(),
        TrajectoryStore(args.teacher).load(),
    )
    manifest = write_comparisons(comparisons, args.output)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
