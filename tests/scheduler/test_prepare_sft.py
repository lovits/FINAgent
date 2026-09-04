import pytest

from tradingagents.scheduler.trajectory import SchedulerStep, SchedulerTrajectory
from training.scheduler.prepare_sft import prepare_sft_dataset


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
            "identical prompt",
            ["<ACT_NEWS>"],
            "<ACT_NEWS>",
            "News Analyst",
            state_after={"news_report": "done"},
        )
    )
    trajectory.provenance = {"task_split": "train"}
    return trajectory


def test_combines_sources_and_deduplicates_identical_transition() -> None:
    static = _trajectory("static-1", "static")
    teacher = _trajectory("teacher-1", "teacher")
    examples, overflow = prepare_sft_dataset(
        [static],
        [teacher],
        verified_ids={"teacher-1"},
        token_counter=len,
    )
    assert len(examples) == 1
    assert examples[0].source == "teacher_verified"
    assert overflow == 0


def test_reports_overflow_from_both_sources() -> None:
    examples, overflow = prepare_sft_dataset(
        [_trajectory("static-1", "static")],
        [_trajectory("teacher-1", "teacher")],
        verified_ids={"teacher-1"},
        token_counter=lambda _: 100,
        max_tokens=50,
    )
    assert examples == []
    assert overflow == 2


def test_includes_audited_teacher_only_tasks_as_separate_source() -> None:
    audited = _trajectory("teacher-audited-1", "teacher")
    audited.task_id = "task-2"
    audited.steps[0].serialized_state = "audited teacher prompt"

    examples, overflow = prepare_sft_dataset(
        [_trajectory("static-1", "static")],
        [_trajectory("teacher-1", "teacher")],
        [audited],
        verified_ids={"teacher-1"},
        token_counter=len,
    )

    assert {example.source for example in examples} == {
        "teacher_verified",
        "teacher_audited",
    }
    assert overflow == 0


def test_rejects_teacher_only_source_that_overlaps_paired_task() -> None:
    with pytest.raises(ValueError, match="already present in paired Teacher"):
        prepare_sft_dataset(
            [_trajectory("static-1", "static")],
            [_trajectory("teacher-1", "teacher")],
            [_trajectory("teacher-audited-1", "teacher")],
            verified_ids={"teacher-1"},
            token_counter=len,
        )


def test_prefers_teacher_label_for_static_teacher_conflict() -> None:
    static = _trajectory("static-1", "static")
    teacher = _trajectory("teacher-1", "teacher")
    teacher.steps[0].selected_action = "<ACT_MARKET>"
    teacher.steps[0].valid_actions.append("<ACT_MARKET>")
    teacher.steps[0].agent_node = "Market Analyst"

    examples, _ = prepare_sft_dataset(
        [static],
        [teacher],
        verified_ids={"teacher-1"},
        token_counter=len,
    )

    assert len(examples) == 1
    assert examples[0].source == "teacher_verified"
    assert examples[0].target_action == "<ACT_MARKET>"


def test_includes_unverified_paired_teacher_as_audited() -> None:
    teacher = _trajectory("teacher-1", "teacher")
    teacher.steps[0].serialized_state = "dynamic teacher prompt"

    examples, _ = prepare_sft_dataset(
        [_trajectory("static-1", "static")],
        [teacher],
        verified_ids=set(),
        token_counter=len,
    )

    assert {example.source for example in examples} == {
        "static",
        "teacher_audited",
    }


def test_rejects_conflicting_labels_from_equal_teacher_sources() -> None:
    first = _trajectory("teacher-audited-1", "teacher")
    first.task_id = "task-2"
    second = _trajectory("teacher-audited-2", "teacher")
    second.task_id = "task-3"
    second.steps[0].selected_action = "<ACT_MARKET>"
    second.steps[0].valid_actions.append("<ACT_MARKET>")
    second.steps[0].agent_node = "Market Analyst"

    with pytest.raises(ValueError, match="equally trusted"):
        prepare_sft_dataset(
            [_trajectory("static-1", "static")],
            [],
            [first, second],
            verified_ids=set(),
            token_counter=len,
        )


def test_rejects_mixed_train_and_validation_sources() -> None:
    static = _trajectory("static-1", "static")
    teacher = _trajectory("teacher-1", "teacher")
    teacher.provenance["task_split"] = "validation"

    with pytest.raises(ValueError, match="mix task splits"):
        prepare_sft_dataset(
            [static],
            [teacher],
            verified_ids={"teacher-1"},
            token_counter=len,
        )
