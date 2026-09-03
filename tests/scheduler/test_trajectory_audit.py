import copy
import json

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.trajectory import SchedulerTrajectory, TrajectoryStep
from training.scheduler.trajectory_audit import (
    audit_scheduler_record,
    audit_static_record,
    build_audit_report,
    load_and_audit,
)


def _serialized(ticker, trade_date, actions):
    return json.dumps(
        {
            "task": {"ticker": ticker, "trade_date": trade_date},
            "valid_actions": actions,
        }
    )


def _static_record():
    ticker = "NVDA"
    trade_date = "2026-08-26"
    return {
        "record_type": "static_langgraph",
        "dataset_version": "v1",
        "action_schema_version": "v1",
        "state_schema_version": "v1",
        "generation_key": "task:static:0:v1",
        "trajectory_id": "trajectory-1",
        "status": "accepted",
        "task": {
            "task_id": "task",
            "ticker": ticker,
            "trade_date": trade_date,
            "split": "train",
        },
        "provenance": {
            "graph_mode": "static",
            "information_cutoff": trade_date,
        },
        "node_steps": [{"step_id": 0}, {"step_id": 1}],
        "scheduler_examples": [
            {
                "step_id": 0,
                "input_text": _serialized(
                    ticker, trade_date, [SchedulerAction.MARKET.value]
                ),
                "valid_actions": [SchedulerAction.MARKET.value],
                "target_action": SchedulerAction.MARKET.value,
                "source_node": "Market Analyst",
                "action_valid": True,
            },
            {
                "step_id": 1,
                "input_text": _serialized(
                    ticker, trade_date, [SchedulerAction.STOP.value]
                ),
                "valid_actions": [SchedulerAction.STOP.value],
                "target_action": SchedulerAction.STOP.value,
                "source_node": "END",
                "action_valid": True,
            },
        ],
        "final_outputs": {"trader_action": "Buy", "portfolio_rating": "Buy"},
        "labels": {"completed": True, "data_sparse": False},
        "quality_control": {
            "static_actions_valid": True,
            "failure_reason": None,
            "human_review": "approved",
        },
    }


def test_valid_static_record_is_eligible_for_sft():
    result = audit_static_record(_static_record())

    assert result.eligible_for_sft is True
    assert result.has_errors is False


def test_static_record_with_wrong_terminal_action_is_blocked():
    record = copy.deepcopy(_static_record())
    record["scheduler_examples"].pop()

    result = audit_static_record(record)

    assert result.eligible_for_sft is False
    assert {issue.code for issue in result.issues} >= {
        "terminal_stop_missing",
        "terminal_stop_count",
    }


def test_valid_learned_trajectory_is_eligible_for_sft():
    trajectory = SchedulerTrajectory(
        trajectory_id="learned-1",
        task_id="task",
        run_id="run",
        mode="learned",
        policy_id="teacher",
        ticker="NVDA",
        trade_date="2026-08-26",
        verifier_version="v1",
        status="accepted",
        trader_result="FINAL TRANSACTION PROPOSAL: **BUY**",
        final_decision="**Rating**: Buy",
        metadata={"task_split": "train"},
    )
    for step_id, action in enumerate((SchedulerAction.MARKET, SchedulerAction.STOP)):
        trajectory.steps.append(
            TrajectoryStep(
                step_id=step_id,
                serialized_state="{}",
                valid_actions=[action.value],
                selected_action=action.value,
                policy_id="teacher",
            )
        )

    result = audit_scheduler_record(trajectory.to_dict())

    assert result.eligible_for_sft is True


def test_file_audit_detects_duplicate_trajectory_ids(tmp_path):
    path = tmp_path / "accepted.jsonl"
    payload = json.dumps(_static_record()) + "\n"
    path.write_text(payload + payload, encoding="utf-8")

    audited = load_and_audit([path])
    report = build_audit_report(audited)

    assert report["records"] == 2
    assert report["eligible_for_sft"] == 1
    assert report["blocking_records"] == 1
    assert report["issue_counts"]["error:trajectory_id_duplicate"] == 1
