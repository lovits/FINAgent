"""Extend a canonical SFT source bundle with new accepted trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .assemble_sft_sources import TrajectoryKey, load_accepted_roots
from .pair import PairComparison, pair_trajectories
from .pair_dataset import load_comparisons, write_comparisons

EXTENSION_VERSION = "scheduler-sft-source-extension-v1"


def extend_sft_sources(
    base_static: Iterable[SchedulerTrajectory],
    base_paired_teacher: Iterable[SchedulerTrajectory],
    base_audited_teacher: Iterable[SchedulerTrajectory],
    base_comparisons: Iterable[PairComparison],
    extra_paired: Iterable[SchedulerTrajectory],
    extra_teacher_only: Iterable[SchedulerTrajectory],
) -> tuple[
    list[SchedulerTrajectory],
    list[SchedulerTrajectory],
    list[SchedulerTrajectory],
    list[PairComparison],
]:
    base_static = list(base_static)
    base_paired_teacher = list(base_paired_teacher)
    base_audited_teacher = list(base_audited_teacher)
    base_comparisons = list(base_comparisons)
    base = [*base_static, *base_paired_teacher, *base_audited_teacher]
    extra_paired = list(extra_paired)
    extra_teacher_only = list(extra_teacher_only)
    _reject_task_overlap(base, [*extra_paired, *extra_teacher_only])
    extra_static = _by_key(extra_paired, "static")
    extra_teacher = _by_key(extra_paired, "teacher")
    common = sorted(set(extra_static) & set(extra_teacher))
    external_audited = _by_key(extra_teacher_only, "teacher")
    if set(external_audited) & (set(extra_static) | set(extra_teacher)):
        raise ValueError("Teacher-only extension overlaps paired extension tasks")
    static = [*base_static, *[extra_static[key] for key in sorted(extra_static)]]
    paired_teacher = [*base_paired_teacher, *[extra_teacher[key] for key in common]]
    unpaired_teacher = [extra_teacher[key] for key in sorted(set(extra_teacher) - set(common))]
    audited_teacher = [
        *base_audited_teacher,
        *unpaired_teacher,
        *[external_audited[key] for key in sorted(external_audited)],
    ]
    comparisons = [
        *base_comparisons,
        *[pair_trajectories(extra_static[key], extra_teacher[key]) for key in common],
    ]
    _validate_partition(paired_teacher, audited_teacher)
    return static, paired_teacher, audited_teacher, comparisons


def _by_key(
    trajectories: Iterable[SchedulerTrajectory], expected_mode: str
) -> dict[TrajectoryKey, SchedulerTrajectory]:
    values = {}
    for trajectory in trajectories:
        if trajectory.mode != expected_mode:
            continue
        key = (trajectory.task_id, trajectory.data_snapshot_id)
        if key in values:
            raise ValueError(f"duplicate accepted {expected_mode} trajectory for {key}")
        values[key] = trajectory
    return values


def _task_keys(trajectories: Iterable[SchedulerTrajectory]) -> set[TrajectoryKey]:
    return {(trajectory.task_id, trajectory.data_snapshot_id) for trajectory in trajectories}


def _reject_task_overlap(
    base: Iterable[SchedulerTrajectory], extra: Iterable[SchedulerTrajectory]
) -> None:
    overlap = _task_keys(base) & _task_keys(extra)
    if overlap:
        raise ValueError(f"extension overlaps existing SFT tasks: {len(overlap)}")


def _validate_partition(
    paired_teacher: Iterable[SchedulerTrajectory],
    audited_teacher: Iterable[SchedulerTrajectory],
) -> None:
    overlap = _task_keys(paired_teacher) & _task_keys(audited_teacher)
    if overlap:
        raise ValueError(f"paired and audited Teacher sources overlap: {len(overlap)}")


def write_extended_sources(
    output_dir: str | Path,
    static: Sequence[SchedulerTrajectory],
    paired_teacher: Sequence[SchedulerTrajectory],
    audited_teacher: Sequence[SchedulerTrajectory],
    comparisons: Sequence[PairComparison],
) -> dict[str, Any]:
    output = Path(output_dir)
    _write_trajectories(output / "static.accepted.jsonl", static)
    _write_trajectories(output / "teacher-paired.accepted.jsonl", paired_teacher)
    _write_trajectories(output / "teacher-audited.accepted.jsonl", audited_teacher)
    write_comparisons(comparisons, output / "comparisons.jsonl")
    manifest = _manifest(static, paired_teacher, audited_teacher, comparisons)
    write_json_atomic(output / "extension_manifest.json", manifest)
    return manifest


def _write_trajectories(path: Path, trajectories: Sequence[SchedulerTrajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(
                json.dumps(trajectory.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def _manifest(
    static: Sequence[SchedulerTrajectory],
    paired_teacher: Sequence[SchedulerTrajectory],
    audited_teacher: Sequence[SchedulerTrajectory],
    comparisons: Sequence[PairComparison],
) -> dict[str, Any]:
    trajectories = [*static, *paired_teacher, *audited_teacher]
    return {
        "schema_version": EXTENSION_VERSION,
        "accepted_trajectory_count": len(trajectories),
        "task_count": len(_task_keys(trajectories)),
        "source_trajectory_counts": {
            "static": len(static),
            "teacher_paired": len(paired_teacher),
            "teacher_audited": len(audited_teacher),
        },
        "pair_status_counts": dict(Counter(item.pair_status for item in comparisons)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--paired-root", action="append", required=True)
    parser.add_argument("--teacher-only-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    base = Path(args.base_dir)
    values = extend_sft_sources(
        TrajectoryStore(base / "static.accepted.jsonl").load(),
        TrajectoryStore(base / "teacher-paired.accepted.jsonl").load(),
        TrajectoryStore(base / "teacher-audited.accepted.jsonl").load(),
        load_comparisons(base / "comparisons.jsonl"),
        load_accepted_roots(args.paired_root),
        load_accepted_roots(args.teacher_only_root),
    )
    manifest = write_extended_sources(args.output_dir, *values)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
