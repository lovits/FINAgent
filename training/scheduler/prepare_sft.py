"""Build split-aware SFT JSONL from Static and audited Teacher traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable

from tradingagents.scheduler.store import TrajectoryStore
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .build_sft import (
    SFTExample,
    build_sft_examples,
    verified_teacher_ids,
    write_sft_dataset,
)
from .model import scheduler_input_ids
from .pair_dataset import load_comparisons

SOURCE_PRIORITY = {
    "static": 0,
    "teacher_audited": 1,
    "teacher_verified": 2,
}


def prepare_sft_dataset(
    static: Iterable[SchedulerTrajectory],
    teacher: Iterable[SchedulerTrajectory],
    teacher_audited: Iterable[SchedulerTrajectory] = (),
    *,
    verified_ids: set[str],
    token_counter: Callable[[str], int],
    max_tokens: int = 32768,
) -> tuple[list[SFTExample], int]:
    static = list(static)
    teacher = list(teacher)
    teacher_audited = list(teacher_audited)
    source_task_split(static, teacher, teacher_audited)
    paired_teacher_tasks = {
        (trajectory.task_id, trajectory.data_snapshot_id) for trajectory in teacher
    }
    audited_teacher_tasks = {
        (trajectory.task_id, trajectory.data_snapshot_id)
        for trajectory in teacher_audited
    }
    overlap = paired_teacher_tasks & audited_teacher_tasks
    if overlap:
        raise ValueError(
            "teacher_audited contains tasks already present in paired Teacher "
            f"data: {len(overlap)}"
        )
    static_examples, static_overflow = build_sft_examples(
        static,
        source="static",
        token_counter=token_counter,
        max_tokens=max_tokens,
    )
    teacher_examples, teacher_overflow = build_sft_examples(
        teacher,
        source="teacher_verified",
        token_counter=token_counter,
        max_tokens=max_tokens,
        verified_teacher_ids=verified_ids,
    )
    unverified_teacher = [
        trajectory
        for trajectory in teacher
        if trajectory.trajectory_id not in verified_ids
    ]
    audited_examples, audited_overflow = build_sft_examples(
        [*unverified_teacher, *teacher_audited],
        source="teacher_audited",
        token_counter=token_counter,
        max_tokens=max_tokens,
    )
    return _deduplicate(
        [*static_examples, *teacher_examples, *audited_examples]
    ), (
        static_overflow + teacher_overflow + audited_overflow
    )


def source_task_split(
    static: Iterable[SchedulerTrajectory],
    teacher: Iterable[SchedulerTrajectory],
    teacher_audited: Iterable[SchedulerTrajectory] = (),
) -> str:
    splits = {
        trajectory.provenance.get("task_split")
        for trajectory in [*static, *teacher, *teacher_audited]
    }
    if not splits:
        raise ValueError("SFT sources are empty")
    if None in splits or "" in splits:
        raise ValueError("SFT source trajectory is missing task_split provenance")
    if len(splits) != 1:
        raise ValueError(f"SFT sources mix task splits: {sorted(splits)}")
    return str(next(iter(splits)))


def _deduplicate(examples: Iterable[SFTExample]) -> list[SFTExample]:
    values: dict[str, SFTExample] = {}
    for example in examples:
        existing = values.get(example.input_text)
        if existing is None:
            values[example.input_text] = example
            continue
        existing_priority = SOURCE_PRIORITY[existing.source]
        candidate_priority = SOURCE_PRIORITY[example.source]
        if existing.target_action != example.target_action and (
            existing_priority == candidate_priority
        ):
            raise ValueError("conflicting actions from equally trusted sources")
        if candidate_priority > existing_priority:
            values[example.input_text] = example
    return list(values.values())


def qwen_token_counter(model_id: str, revision: str | None = None) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def count(input_text: str) -> int:
        return len(scheduler_input_ids(tokenizer, input_text))

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--teacher-audited")
    parser.add_argument("--comparisons", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision")
    parser.add_argument("--max-tokens", type=int, default=32768)
    args = parser.parse_args()

    static = TrajectoryStore(args.static).load()
    teacher = TrajectoryStore(args.teacher).load()
    teacher_audited = (
        TrajectoryStore(args.teacher_audited).load()
        if args.teacher_audited
        else []
    )
    comparisons = load_comparisons(args.comparisons)
    task_split = source_task_split(static, teacher, teacher_audited)
    examples, overflow = prepare_sft_dataset(
        static,
        teacher,
        teacher_audited,
        verified_ids=verified_teacher_ids(comparisons),
        token_counter=qwen_token_counter(args.model_id, args.revision),
        max_tokens=args.max_tokens,
    )
    run_ids = sorted(
        {
            trajectory.run_id
            for trajectory in [*static, *teacher, *teacher_audited]
        }
    )
    manifest = write_sft_dataset(
        examples,
        args.output,
        overflow_count=overflow,
        source_run_ids=run_ids,
        task_split=task_split,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
