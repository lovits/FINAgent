"""Deterministic audit gates for Static and learned scheduler trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from tradingagents.scheduler.actions import (
    ACTION_SCHEMA_VERSION,
    NODE_BY_ACTION,
    SchedulerAction,
    parse_action,
)
from tradingagents.scheduler.state_serializer import STATE_SCHEMA_VERSION
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .reward import parse_portfolio_rating, parse_trader_action

AUDIT_VERSION = "v1"
Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class AuditIssue:
    severity: Severity
    code: str
    message: str


@dataclass
class TrajectoryAuditResult:
    source_file: str
    line_number: int
    record_type: str
    trajectory_id: str | None
    task_id: str | None
    status: str
    eligible_for_sft: bool = False
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _add(
    result: TrajectoryAuditResult,
    severity: Severity,
    code: str,
    message: str,
) -> None:
    result.issues.append(AuditIssue(severity, code, message))


def _result_for(
    record: Mapping[str, Any], source_file: str, line_number: int, record_type: str
) -> TrajectoryAuditResult:
    task = record.get("task") if isinstance(record.get("task"), Mapping) else {}
    return TrajectoryAuditResult(
        source_file=source_file,
        line_number=line_number,
        record_type=record_type,
        trajectory_id=record.get("trajectory_id"),
        task_id=task.get("task_id") or record.get("task_id"),
        status=str(record.get("status") or "unknown"),
    )


def _parse_actions(
    values: Any,
    result: TrajectoryAuditResult,
    *,
    step_id: int,
) -> list[SchedulerAction]:
    if not isinstance(values, list) or not values:
        _add(result, "error", "valid_actions_missing", f"step {step_id} has no valid actions")
        return []
    try:
        return [parse_action(value) for value in values]
    except (TypeError, ValueError) as exc:
        _add(result, "error", "valid_action_invalid", f"step {step_id}: {exc}")
        return []


def _audit_static_steps(
    record: Mapping[str, Any], result: TrajectoryAuditResult
) -> None:
    examples = record.get("scheduler_examples")
    if not isinstance(examples, list) or not examples:
        _add(result, "error", "scheduler_examples_missing", "no scheduler examples")
        return

    actions: list[SchedulerAction | None] = []
    task = record["task"]
    for expected_step, step in enumerate(examples):
        if not isinstance(step, Mapping):
            _add(result, "error", "scheduler_step_invalid", f"step {expected_step} is not an object")
            actions.append(None)
            continue
        if step.get("step_id") != expected_step:
            _add(result, "error", "scheduler_step_gap", f"expected step {expected_step}")
        valid = _parse_actions(step.get("valid_actions"), result, step_id=expected_step)
        try:
            action = parse_action(step.get("target_action"))
        except (TypeError, ValueError) as exc:
            _add(result, "error", "target_action_invalid", f"step {expected_step}: {exc}")
            actions.append(None)
            continue
        actions.append(action)
        if action not in valid:
            _add(result, "error", "target_action_masked", f"step {expected_step} target is masked")
        if step.get("action_valid") is not True:
            _add(result, "error", "action_not_verified", f"step {expected_step} was not verified")
        expected_node = "END" if action is SchedulerAction.STOP else NODE_BY_ACTION[action]
        if step.get("source_node") != expected_node:
            _add(result, "error", "source_node_mismatch", f"step {expected_step} node mismatch")
        _audit_serialized_state(step, task, valid, result, expected_step)

    if actions[-1] is not SchedulerAction.STOP:
        _add(result, "error", "terminal_stop_missing", "last scheduler action is not STOP")
    if sum(action is SchedulerAction.STOP for action in actions) != 1:
        _add(result, "error", "terminal_stop_count", "trajectory must contain exactly one STOP")


def _audit_serialized_state(
    step: Mapping[str, Any],
    task: Mapping[str, Any],
    valid: Sequence[SchedulerAction],
    result: TrajectoryAuditResult,
    step_id: int,
) -> None:
    try:
        payload = json.loads(step.get("input_text"))
    except (TypeError, json.JSONDecodeError) as exc:
        _add(result, "error", "serialized_state_invalid", f"step {step_id}: {exc}")
        return
    serialized_task = payload.get("task") if isinstance(payload, Mapping) else None
    if not isinstance(serialized_task, Mapping):
        _add(result, "error", "serialized_task_missing", f"step {step_id} has no task")
        return
    for task_field in ("ticker", "trade_date"):
        if str(serialized_task.get(task_field)) != str(task.get(task_field)):
            _add(
                result,
                "error",
                "serialized_task_mismatch",
                f"step {step_id} {task_field} mismatch",
            )
    serialized_valid = payload.get("valid_actions")
    if serialized_valid != [action.value for action in valid]:
        _add(result, "error", "serialized_mask_mismatch", f"step {step_id} mask mismatch")


def audit_static_record(
    record: Mapping[str, Any], *, source_file: str = "<memory>", line_number: int = 1
) -> TrajectoryAuditResult:
    result = _result_for(record, source_file, line_number, "static_langgraph")
    task = record.get("task")
    if not isinstance(task, Mapping):
        _add(result, "error", "task_missing", "static record has no task object")
        return result
    for field_name in ("task_id", "ticker", "trade_date", "split"):
        if not task.get(field_name):
            _add(result, "error", "task_field_missing", f"task is missing {field_name}")
    if result.status not in {"accepted", "rejected"}:
        _add(result, "error", "status_invalid", f"invalid static status: {result.status}")
    if record.get("action_schema_version") != ACTION_SCHEMA_VERSION:
        _add(result, "error", "action_schema_mismatch", "unsupported action schema")
    if record.get("state_schema_version") != STATE_SCHEMA_VERSION:
        _add(result, "error", "state_schema_mismatch", "unsupported state schema")
    if not record.get("generation_key"):
        _add(result, "warning", "generation_key_missing", "legacy record cannot be resumed safely")

    provenance = record.get("provenance") or {}
    if provenance.get("graph_mode") != "static":
        _add(result, "error", "graph_mode_mismatch", "record was not produced by static mode")
    if str(provenance.get("information_cutoff")) != str(task.get("trade_date")):
        _add(result, "error", "information_cutoff_mismatch", "cutoff differs from trade date")

    node_steps = record.get("node_steps")
    if not isinstance(node_steps, list) or not node_steps:
        _add(result, "error", "node_steps_missing", "no LangGraph node events were captured")
    elif [step.get("step_id") for step in node_steps] != list(range(len(node_steps))):
        _add(result, "error", "node_step_gap", "LangGraph node step IDs are not contiguous")

    if result.status == "accepted":
        _audit_static_steps(record, result)
        outputs = record.get("final_outputs") or {}
        if outputs.get("trader_action") not in {"Buy", "Hold", "Sell"}:
            _add(result, "error", "trader_action_missing", "final Trader action is invalid")
        if parse_portfolio_rating(str(outputs.get("portfolio_rating") or "")) is None:
            _add(result, "error", "portfolio_rating_missing", "final rating is invalid")
        labels = record.get("labels") or {}
        if labels.get("completed") is not True:
            _add(result, "error", "completion_label_false", "accepted record is not complete")
        quality = record.get("quality_control") or {}
        if quality.get("static_actions_valid") is not True or quality.get("failure_reason"):
            _add(result, "error", "quality_control_failed", "accepted record failed its runtime gate")
        if labels.get("data_sparse"):
            _add(result, "warning", "data_sparse", "one or more selected data sources were sparse")
        if quality.get("human_review") == "pending":
            _add(result, "warning", "human_review_pending", "manual review is still pending")
    else:
        _add(result, "warning", "source_rejected", "record was rejected during collection")

    result.eligible_for_sft = result.status == "accepted" and not result.has_errors
    return result


def audit_scheduler_record(
    record: Mapping[str, Any], *, source_file: str = "<memory>", line_number: int = 1
) -> TrajectoryAuditResult:
    result = _result_for(record, source_file, line_number, "scheduler_trajectory")
    try:
        trajectory = SchedulerTrajectory.from_dict(dict(record))
    except (TypeError, ValueError) as exc:
        _add(result, "error", "trajectory_schema_invalid", str(exc))
        return result
    if trajectory.status not in {"accepted", "rejected", "failed", "fallback"}:
        _add(result, "error", "status_invalid", f"non-terminal status: {trajectory.status}")
    if trajectory.status == "accepted":
        if not trajectory.steps or trajectory.steps[-1].selected_action != SchedulerAction.STOP.value:
            _add(result, "error", "terminal_stop_missing", "last scheduler action is not STOP")
        if parse_trader_action(trajectory.trader_result) is None:
            _add(result, "error", "trader_action_missing", "final Trader action is invalid")
        if parse_portfolio_rating(trajectory.final_decision) is None:
            _add(result, "error", "portfolio_rating_missing", "final rating is invalid")
        if not trajectory.verifier_version:
            _add(result, "error", "verifier_missing", "accepted trajectory has no verifier version")
        for step in trajectory.steps:
            try:
                json.loads(step.serialized_state)
            except json.JSONDecodeError as exc:
                _add(result, "error", "serialized_state_invalid", f"step {step.step_id}: {exc}")
        if not trajectory.metadata.get("task_split"):
            _add(result, "error", "task_split_missing", "SFT split provenance is missing")
    else:
        _add(result, "warning", "source_rejected", "record was rejected during collection")
    result.eligible_for_sft = trajectory.status == "accepted" and not result.has_errors
    return result


def audit_record(
    record: Mapping[str, Any], *, source_file: str = "<memory>", line_number: int = 1
) -> TrajectoryAuditResult:
    if record.get("record_type") == "static_langgraph":
        return audit_static_record(record, source_file=source_file, line_number=line_number)
    return audit_scheduler_record(record, source_file=source_file, line_number=line_number)


def load_and_audit(
    paths: Sequence[str | Path],
) -> list[tuple[dict[str, Any], TrajectoryAuditResult]]:
    audited = []
    seen_ids: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError("trajectory record must be an object")
                    result = audit_record(
                        record,
                        source_file=str(path),
                        line_number=line_number,
                    )
                except (TypeError, json.JSONDecodeError) as exc:
                    record = {}
                    result = _result_for(record, str(path), line_number, "invalid")
                    _add(result, "error", "json_invalid", str(exc))
                if result.trajectory_id:
                    if result.trajectory_id in seen_ids:
                        _add(result, "error", "trajectory_id_duplicate", "duplicate trajectory ID")
                        result.eligible_for_sft = False
                    seen_ids.add(result.trajectory_id)
                audited.append((record, result))
    return audited


def build_audit_report(
    audited: Sequence[tuple[dict[str, Any], TrajectoryAuditResult]],
) -> dict[str, Any]:
    results = [result for _, result in audited]
    issue_counts = Counter(
        f"{issue.severity}:{issue.code}" for result in results for issue in result.issues
    )
    return {
        "audit_version": AUDIT_VERSION,
        "records": len(results),
        "eligible_for_sft": sum(result.eligible_for_sft for result in results),
        "blocking_records": sum(
            result.has_errors and result.status != "rejected" for result in results
        ),
        "status_counts": dict(Counter(result.status for result in results)),
        "record_type_counts": dict(Counter(result.record_type for result in results)),
        "issue_counts": dict(sorted(issue_counts.items())),
        "results": [result.to_dict() for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_audit_report(load_and_audit(args.input))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("records", "eligible_for_sft", "blocking_records")}))
    if report["blocking_records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
