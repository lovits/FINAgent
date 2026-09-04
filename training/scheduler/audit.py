"""Deterministic structural and completion audit for scheduler trajectories."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.scheduler.actions import SchedulerAction, node_for_action, parse_action
from tradingagents.scheduler.trajectory import SchedulerTrajectory

AUDITOR_VERSION = "scheduler-audit-v2"
_REPORT_BY_ANALYST = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}
_TRADER_ACTION = re.compile(
    r"(?:\*{0,2}Action\*{0,2}|FINAL\s+TRANSACTION\s+PROPOSAL)\s*:\s*\*{0,2}"
    r"(BUY|HOLD|SELL)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AuditRecord:
    trajectory_id: str
    audit_status: str
    checks: dict[str, bool]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    auditor_version: str = AUDITOR_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def parse_trader_action(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    match = _TRADER_ACTION.search(text)
    if match:
        return match.group(1).title()
    return None


def parse_portfolio_rating(text: object) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return parse_rating(text, default="") or None


def audit_trajectory(trajectory: SchedulerTrajectory) -> AuditRecord:
    errors: list[str] = []
    warnings: list[str] = []
    checks = {
        "schema_valid": trajectory.schema_version == "scheduler-trajectory-v1",
        "steps_contiguous": all(
            step.step_id == index for index, step in enumerate(trajectory.steps)
        ),
        "actions_legal": all(
            step.selected_action in step.valid_actions for step in trajectory.steps
        ),
        "nodes_match_actions": _nodes_match_actions(trajectory),
        "state_progress_valid": _state_progress_valid(trajectory),
        "selected_analyst_reports_complete": _selected_reports_complete(trajectory),
        "completion_valid": _completion_valid(trajectory),
        "task_snapshot_match": bool(
            trajectory.task_id and trajectory.ticker and trajectory.trade_date
        ),
    }
    errors.extend(name for name, passed in checks.items() if not passed)
    if trajectory.execution_status != "completed":
        errors.append(f"execution_status:{trajectory.execution_status}")
    if any(step.error for step in trajectory.steps):
        errors.append("step_error")

    status = "rejected" if errors else ("warning" if warnings else "accepted")
    record = AuditRecord(
        trajectory.trajectory_id,
        status,
        checks,
        tuple(dict.fromkeys(errors)),
        tuple(warnings),
    )
    trajectory.audit_status = status
    trajectory.audit = record.to_dict()
    return record


def _nodes_match_actions(trajectory: SchedulerTrajectory) -> bool:
    for step in trajectory.steps:
        action = parse_action(step.selected_action)
        expected = None if action is SchedulerAction.STOP else node_for_action(action)
        if step.agent_node != expected:
            return False
    return True


def _state_progress_valid(trajectory: SchedulerTrajectory) -> bool:
    return all(
        step.selected_action == SchedulerAction.STOP.value
        or step.state_before != step.state_after
        for step in trajectory.steps
    )


def _selected_reports_complete(trajectory: SchedulerTrajectory) -> bool:
    selected = trajectory.provenance.get("selected_analysts")
    if selected is None:
        return True
    if not isinstance(selected, (list, tuple)) or not selected:
        return False
    try:
        report_fields = [_REPORT_BY_ANALYST[str(key)] for key in selected]
    except KeyError:
        return False
    final_state = trajectory.steps[-1].state_before if trajectory.steps else {}
    return all(str(final_state.get(field) or "").strip() for field in report_fields)


def _completion_valid(trajectory: SchedulerTrajectory) -> bool:
    if not trajectory.steps:
        return False
    stop_count = sum(
        step.selected_action == SchedulerAction.STOP.value for step in trajectory.steps
    )
    outputs = trajectory.final_outputs
    return (
        stop_count == 1
        and trajectory.steps[-1].selected_action == SchedulerAction.STOP.value
        and bool(str(outputs.get("investment_plan") or "").strip())
        and parse_trader_action(outputs.get("trader_investment_plan")) is not None
        and parse_portfolio_rating(outputs.get("final_trade_decision")) is not None
    )
