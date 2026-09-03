"""Shared summaries for persisted scheduler trajectory datasets."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from tradingagents.scheduler.trajectory import SchedulerTrajectory


def summarize_trajectories(
    trajectories: Iterable[SchedulerTrajectory],
) -> dict[str, int]:
    values = list(trajectories)
    execution = Counter(value.execution_status for value in values)
    audits = Counter(value.audit_status for value in values)
    return {
        "total": len(values),
        "completed": execution["completed"],
        "failed": execution["failed"],
        "fallback": execution["fallback"],
        "budget_exhausted": execution["budget_exhausted"],
        "context_overflow": execution["context_overflow"],
        "accepted": audits["accepted"] + audits["warning"],
        "warning": audits["warning"],
        "rejected": audits["rejected"],
        "pending_audit": audits["pending"],
    }
