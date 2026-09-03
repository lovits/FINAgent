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
    assert examples[0].source == "static"
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


def test_rejects_conflicting_labels_for_identical_input() -> None:
    static = _trajectory("static-1", "static")
    teacher = _trajectory("teacher-1", "teacher")
    teacher.steps[0].selected_action = "<ACT_MARKET>"
    teacher.steps[0].valid_actions.append("<ACT_MARKET>")
    teacher.steps[0].agent_node = "Market Analyst"

    with pytest.raises(ValueError, match="conflicting actions"):
        prepare_sft_dataset(
            [static],
            [teacher],
            verified_ids={"teacher-1"},
            token_counter=len,
        )
