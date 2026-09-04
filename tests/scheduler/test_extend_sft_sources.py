import pytest

from tradingagents.scheduler.trajectory import SchedulerTrajectory
from training.scheduler.extend_sft_sources import extend_sft_sources
from training.scheduler.pair import pair_trajectories


def _trajectory(task_id: str, mode: str) -> SchedulerTrajectory:
    trajectory = SchedulerTrajectory(
        trajectory_id=f"{mode}-{task_id}",
        run_id=f"run-{mode}-{task_id}",
        task_id=task_id,
        mode=mode,  # type: ignore[arg-type]
        policy_id=f"{mode}-policy",
        ticker=task_id,
        trade_date="2026-01-05",
        data_snapshot_id=f"snapshot-{task_id}",
        execution_status="completed",
        audit_status="accepted",
    )
    trajectory.final_outputs = {
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
        "final_trade_decision": "**Rating**: Hold",
    }
    trajectory.provenance = {
        "task_dataset_version": "scheduler-tasks-v2",
        "task_split": "train",
        "information_cutoff": "2026-01-05T23:59:59Z",
        "selected_analysts": ["market"],
        "research_depth": "shallow",
        "output_language": "Chinese",
        "expert_config_hash": "same",
        "scheduler_max_steps": 16,
    }
    return trajectory


def test_extends_bundle_with_paired_and_unpaired_accepted_sources() -> None:
    base_static = [_trajectory("BASE_PAIR", "static")]
    base_teacher = [_trajectory("BASE_PAIR", "teacher")]
    base_audited = [_trajectory("BASE_ONLY", "teacher")]
    extra_paired = [
        _trajectory("NEW_PAIR", "static"),
        _trajectory("NEW_PAIR", "teacher"),
        _trajectory("STATIC_ONLY", "static"),
        _trajectory("TEACHER_ONLY", "teacher"),
    ]
    extra_teacher_only = [_trajectory("EXTERNAL_ONLY", "teacher")]

    static, paired, audited, comparisons = extend_sft_sources(
        base_static,
        base_teacher,
        base_audited,
        [pair_trajectories(base_static[0], base_teacher[0])],
        extra_paired,
        extra_teacher_only,
    )

    assert len(static) == 3
    assert len(paired) == 2
    assert len(audited) == 3
    assert len(comparisons) == 2


def test_rejects_extension_task_overlap_with_base_bundle() -> None:
    with pytest.raises(ValueError, match="overlaps existing"):
        extend_sft_sources(
            [_trajectory("SAME", "static")],
            [],
            [],
            [],
            [_trajectory("SAME", "teacher")],
            [],
        )
