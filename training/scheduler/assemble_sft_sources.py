"""Assemble exact Static/Teacher trajectory quotas for SFT preparation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .pair import PairComparison, pair_trajectories
from .pair_dataset import write_comparisons

ASSEMBLY_VERSION = "scheduler-sft-source-assembly-v1"
TrajectoryKey = tuple[str, str | None]


def load_accepted_roots(roots: Iterable[str | Path]) -> list[SchedulerTrajectory]:
    trajectories = []
    for root in roots:
        for path in sorted(Path(root).rglob("accepted.jsonl")):
            trajectories.extend(TrajectoryStore(path).load())
    return trajectories


def assemble_sft_sources(
    paired: Iterable[SchedulerTrajectory],
    teacher_only: Iterable[SchedulerTrajectory],
    *,
    paired_task_target: int,
    teacher_trajectory_target: int,
) -> tuple[list[SchedulerTrajectory], list[SchedulerTrajectory], list[SchedulerTrajectory]]:
    if paired_task_target < 1 or teacher_trajectory_target < paired_task_target:
        raise ValueError("invalid Static/Teacher trajectory targets")
    paired = list(paired)
    static = _unique_mode(paired, "static")
    paired_teacher = _unique_mode(paired, "teacher")
    common = sorted(set(static) & set(paired_teacher))
    if len(common) < paired_task_target:
        raise ValueError(f"only {len(common)} accepted paired tasks; required {paired_task_target}")
    selected_keys = common[:paired_task_target]
    selected_static = [static[key] for key in selected_keys]
    selected_paired_teacher = [paired_teacher[key] for key in selected_keys]
    audited_target = teacher_trajectory_target - paired_task_target
    audited = _teacher_only_candidates(teacher_only, blocked=set(static) | set(paired_teacher))
    if len(audited) < audited_target:
        raise ValueError(f"only {len(audited)} Teacher-only tasks; required {audited_target}")
    return selected_static, selected_paired_teacher, audited[:audited_target]


def _unique_mode(
    trajectories: Iterable[SchedulerTrajectory], expected_mode: str
) -> dict[TrajectoryKey, SchedulerTrajectory]:
    values: dict[TrajectoryKey, SchedulerTrajectory] = {}
    for trajectory in trajectories:
        if trajectory.mode != expected_mode:
            continue
        key = (trajectory.task_id, trajectory.data_snapshot_id)
        existing = values.get(key)
        if existing is not None and existing.trajectory_id != trajectory.trajectory_id:
            raise ValueError(f"multiple accepted {expected_mode} trajectories for {key}")
        values[key] = trajectory
    return values


def _teacher_only_candidates(
    trajectories: Iterable[SchedulerTrajectory], *, blocked: set[TrajectoryKey]
) -> list[SchedulerTrajectory]:
    values = _unique_mode(trajectories, "teacher")
    overlap = set(values) & blocked
    if overlap:
        raise ValueError(f"Teacher-only roots overlap paired tasks: {len(overlap)}")
    return [values[key] for key in sorted(values)]


def write_assembled_sources(
    output_dir: str | Path,
    static: Sequence[SchedulerTrajectory],
    paired_teacher: Sequence[SchedulerTrajectory],
    audited_teacher: Sequence[SchedulerTrajectory],
) -> dict[str, Any]:
    output = Path(output_dir)
    _write_trajectories(output / "static.accepted.jsonl", static)
    _write_trajectories(output / "teacher-paired.accepted.jsonl", paired_teacher)
    _write_trajectories(output / "teacher-audited.accepted.jsonl", audited_teacher)
    comparisons = [
        pair_trajectories(static_item, teacher_item)
        for static_item, teacher_item in zip(static, paired_teacher, strict=True)
    ]
    write_comparisons(comparisons, output / "comparisons.jsonl")
    manifest = _assembly_manifest(static, paired_teacher, audited_teacher, comparisons)
    write_json_atomic(output / "assembly_manifest.json", manifest)
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


def _assembly_manifest(
    static: Sequence[SchedulerTrajectory],
    paired_teacher: Sequence[SchedulerTrajectory],
    audited_teacher: Sequence[SchedulerTrajectory],
    comparisons: Sequence[PairComparison],
) -> dict[str, Any]:
    pair_counts = Counter(comparison.pair_status for comparison in comparisons)
    return {
        "schema_version": ASSEMBLY_VERSION,
        "accepted_trajectory_count": len(static) + len(paired_teacher) + len(audited_teacher),
        "source_trajectory_counts": {
            "static": len(static),
            "teacher_paired": len(paired_teacher),
            "teacher_audited": len(audited_teacher),
        },
        "pair_status_counts": dict(pair_counts),
        "task_count": len({item.task_id for item in [*static, *audited_teacher]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-root", action="append", required=True)
    parser.add_argument("--teacher-only-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paired-task-target", type=int, default=20)
    parser.add_argument("--teacher-trajectory-target", type=int, default=80)
    args = parser.parse_args()
    static, paired_teacher, audited_teacher = assemble_sft_sources(
        load_accepted_roots(args.paired_root),
        load_accepted_roots(args.teacher_only_root),
        paired_task_target=args.paired_task_target,
        teacher_trajectory_target=args.teacher_trajectory_target,
    )
    manifest = write_assembled_sources(args.output_dir, static, paired_teacher, audited_teacher)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
