import pytest

from tradingagents.scheduler.trajectory import SchedulerTrajectory
from training.scheduler.pair_dataset import (
    load_comparisons,
    pair_datasets,
    write_comparisons,
)


def _trajectory(identifier: str, mode: str) -> SchedulerTrajectory:
    trajectory = SchedulerTrajectory(
        identifier,
        "run-1",
        "task-1",
        mode,  # type: ignore[arg-type]
        f"{mode}-v1",
        "AAPL",
        "2026-01-05",
        data_snapshot_id="snapshot-1",
        audit_status="accepted",
    )
    trajectory.final_outputs = {
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
        "final_trade_decision": "**Rating**: Hold",
    }
    trajectory.provenance = {
        "task_dataset_version": "scheduler-tasks-v1",
        "information_cutoff": "2026-01-05T23:59:59Z",
        "expert_config_hash": "expert-config-1",
        "scheduler_max_steps": 16,
    }
    return trajectory


def test_pairs_and_round_trips_comparison_records(tmp_path) -> None:
    comparisons = pair_datasets(
        [_trajectory("static-1", "static")],
        [_trajectory("teacher-1", "teacher")],
    )
    assert comparisons[0].pair_status == "teacher_verified"

    target = tmp_path / "comparison.jsonl"
    manifest = write_comparisons(comparisons, target)
    restored = load_comparisons(target)
    assert restored == comparisons
    assert manifest["status_counts"] == {"teacher_verified": 1}


def test_pairing_rejects_missing_teacher_task() -> None:
    static_one = _trajectory("static-1", "static")
    static_two = _trajectory("static-2", "static")
    static_two.task_id = "task-2"
    static_two.data_snapshot_id = "snapshot-2"

    with pytest.raises(ValueError, match="missing_teacher=1"):
        pair_datasets([static_one, static_two], [_trajectory("teacher-1", "teacher")])
