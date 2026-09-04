import json

from tradingagents.scheduler.trajectory import SchedulerStep, SchedulerTrajectory
from training.scheduler.build_sft import (
    build_sft_examples,
    load_sft_examples,
    verified_teacher_ids,
    write_sft_dataset,
)
from training.scheduler.pair import PairComparison


def _trajectory(identifier: str, mode: str) -> SchedulerTrajectory:
    trajectory = SchedulerTrajectory(
        identifier,
        "run-1",
        "task-1",
        mode,  # type: ignore[arg-type]
        f"{mode}-v1",
        "AAPL",
        "2026-01-05",
        audit_status="accepted",
    )
    trajectory.add_step(
        SchedulerStep(
            0,
            {},
            "short prompt",
            ["<ACT_NEWS>"],
            "<ACT_NEWS>",
            "News Analyst",
            state_after={"news_report": "done"},
        )
    )
    trajectory.add_step(
        SchedulerStep(
            1,
            {},
            "x" * 40,
            ["<ACT_STOP>"],
            "<ACT_STOP>",
            None,
        )
    )
    return trajectory


def test_builds_only_verified_teacher_examples_and_rejects_overflow() -> None:
    trajectory = _trajectory("teacher-1", "teacher")
    examples, overflow = build_sft_examples(
        [trajectory],
        source="teacher_verified",
        token_counter=len,
        max_tokens=20,
        verified_teacher_ids={"teacher-1"},
    )
    assert len(examples) == 1
    assert examples[0].target_action == "<ACT_NEWS>"
    assert overflow == 1

    missing, _ = build_sft_examples(
        [trajectory],
        source="teacher_verified",
        token_counter=len,
        verified_teacher_ids=set(),
    )
    assert missing == []


def test_builds_structurally_audited_unpaired_teacher_examples() -> None:
    trajectory = _trajectory("teacher-audited-1", "teacher")
    examples, overflow = build_sft_examples(
        [trajectory],
        source="teacher_audited",
        token_counter=len,
    )

    assert len(examples) == 2
    assert {example.source for example in examples} == {"teacher_audited"}
    assert overflow == 0


def test_verified_teacher_ids_uses_pair_status() -> None:
    comparison = PairComparison(
        "task-1",
        "snapshot-1",
        "static-1",
        "teacher-1",
        True,
        0,
        1.0,
        "teacher_verified",
        (),
    )
    assert verified_teacher_ids([comparison]) == {"teacher-1"}


def test_writes_and_loads_sft_dataset_with_manifest(tmp_path) -> None:
    examples, _ = build_sft_examples(
        [_trajectory("static-1", "static")],
        source="static",
        token_counter=len,
    )
    target = tmp_path / "train.jsonl"
    manifest = write_sft_dataset(examples, target, source_run_ids=("run-1",))

    assert len(load_sft_examples(target)) == 2
    assert manifest["sample_count"] == 2
    saved = json.loads((tmp_path / "train.manifest.json").read_text())
    assert saved["source_counts"] == {"static": 2}
