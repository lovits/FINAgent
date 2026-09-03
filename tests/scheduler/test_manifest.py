from tradingagents.scheduler.trajectory import SchedulerTrajectory
from training.scheduler.manifest import summarize_trajectories


def _trajectory(identifier: str, execution_status: str, audit_status: str):
    return SchedulerTrajectory(
        identifier,
        "run-1",
        identifier,
        "static",
        "static-v1",
        "AAPL",
        "2026-01-05",
        execution_status=execution_status,  # type: ignore[arg-type]
        audit_status=audit_status,  # type: ignore[arg-type]
    )


def test_manifest_summary_keeps_execution_and_audit_axes_separate() -> None:
    summary = summarize_trajectories(
        (
            _trajectory("accepted", "completed", "accepted"),
            _trajectory("warning", "completed", "warning"),
            _trajectory("failed", "failed", "rejected"),
        )
    )

    assert summary["total"] == 3
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["accepted"] == 2
    assert summary["warning"] == 1
    assert summary["rejected"] == 1
