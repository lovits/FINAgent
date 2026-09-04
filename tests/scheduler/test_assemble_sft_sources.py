import pytest

from tradingagents.scheduler.trajectory import SchedulerTrajectory
from training.scheduler.assemble_sft_sources import (
    assemble_sft_sources,
    write_assembled_sources,
)


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


def test_assembles_exact_static_teacher_quota(tmp_path) -> None:
    paired = [
        _trajectory(task_id, mode)
        for task_id in ("PAIR_A", "PAIR_B", "PAIR_C")
        for mode in ("static", "teacher")
    ]
    teacher_only = [_trajectory(f"ONLY_{index}", "teacher") for index in range(4)]

    static, paired_teacher, audited_teacher = assemble_sft_sources(
        paired,
        teacher_only,
        paired_task_target=2,
        teacher_trajectory_target=4,
    )
    manifest = write_assembled_sources(tmp_path, static, paired_teacher, audited_teacher)

    assert len(static) == 2
    assert len(paired_teacher) + len(audited_teacher) == 4
    assert manifest["accepted_trajectory_count"] == 6
    assert manifest["pair_status_counts"] == {"teacher_verified": 2}


def test_rejects_teacher_only_overlap_with_paired_tasks() -> None:
    paired = [_trajectory("PAIR_A", mode) for mode in ("static", "teacher")]

    with pytest.raises(ValueError, match="overlap paired tasks"):
        assemble_sft_sources(
            paired,
            [_trajectory("PAIR_A", "teacher")],
            paired_task_target=1,
            teacher_trajectory_target=2,
        )
