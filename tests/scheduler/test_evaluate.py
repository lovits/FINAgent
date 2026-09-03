import pytest

from tradingagents.scheduler.trajectory import (
    ExecutionCost,
    SchedulerStep,
    SchedulerTrajectory,
)
from training.scheduler.evaluate import evaluate_sets


def _trajectory(identifier: str, mode: str, actions: list[tuple[str, str | None]]):
    trajectory = SchedulerTrajectory(
        identifier,
        "run-1",
        "task-1",
        mode,  # type: ignore[arg-type]
        f"{mode}-v1",
        "AAPL",
        "2026-01-05",
        data_snapshot_id="snapshot-1",
    )
    for index, (action, node) in enumerate(actions):
        before = {"value": index}
        after = before if action == "<ACT_STOP>" else {"value": index + 1}
        trajectory.add_step(
            SchedulerStep(
                index,
                before,
                f"prompt-{index}",
                [action],
                action,
                node,
                state_after=after,
            )
        )
    trajectory.final_outputs = {
        "investment_plan": "Hold",
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
        "final_trade_decision": "**Rating**: Hold",
    }
    trajectory.cost_total = ExecutionCost(agent_calls=len(actions) - 1, tool_calls=2)
    trajectory.execution_status = "completed"
    trajectory.provenance = {
        "task_dataset_version": "scheduler-tasks-v1",
        "information_cutoff": "2026-01-05T23:59:59Z",
        "expert_config_hash": "expert-config-1",
        "scheduler_max_steps": 16,
    }
    return trajectory


def test_evaluation_reports_quality_cost_and_path_dynamics() -> None:
    static = _trajectory(
        "static",
        "static",
        [("<ACT_NEWS>", "News Analyst"), ("<ACT_STOP>", None)],
    )
    learned = _trajectory(
        "learned",
        "learned",
        [("<ACT_MARKET>", "Market Analyst"), ("<ACT_STOP>", None)],
    )
    report = evaluate_sets([static], grpo=[learned])
    metrics = report["modes"]["grpo"]

    assert metrics["completion_rate"] == 1
    assert metrics["trader_match_rate"] == 1
    assert metrics["mean_portfolio_rating_distance"] == 0
    assert metrics["same_as_static_path_rate"] == 0
    assert metrics["unique_paths"] == 1
    assert report["task_alignment"] == {
        "key_fields": ["task_id", "data_snapshot_id"],
        "task_count": 1,
        "strict": True,
    }


def test_evaluation_rejects_missing_candidate_tasks() -> None:
    first = _trajectory(
        "static-1",
        "static",
        [("<ACT_NEWS>", "News Analyst"), ("<ACT_STOP>", None)],
    )
    second = _trajectory(
        "static-2",
        "static",
        [("<ACT_NEWS>", "News Analyst"), ("<ACT_STOP>", None)],
    )
    second.task_id = "task-2"
    second.data_snapshot_id = "snapshot-2"

    with pytest.raises(ValueError, match="not aligned.*missing=1"):
        evaluate_sets([first, second], grpo=[first])


def test_evaluation_rejects_duplicate_task_trajectories() -> None:
    static = _trajectory(
        "static",
        "static",
        [("<ACT_NEWS>", "News Analyst"), ("<ACT_STOP>", None)],
    )
    first = _trajectory(
        "learned-1",
        "learned",
        [("<ACT_NEWS>", "News Analyst"), ("<ACT_STOP>", None)],
    )
    second = _trajectory(
        "learned-2",
        "learned",
        [("<ACT_MARKET>", "Market Analyst"), ("<ACT_STOP>", None)],
    )

    with pytest.raises(ValueError, match="duplicate trajectory"):
        evaluate_sets([static], grpo=[first, second])


def test_evaluation_rejects_explicit_empty_candidate_set() -> None:
    static = _trajectory(
        "static",
        "static",
        [("<ACT_NEWS>", "News Analyst"), ("<ACT_STOP>", None)],
    )

    with pytest.raises(ValueError, match="grpo evaluation set is empty"):
        evaluate_sets([static], grpo=[])


def test_evaluation_rejects_different_expert_configuration() -> None:
    static = _trajectory(
        "static",
        "static",
        [("<ACT_NEWS>", "News Analyst"), ("<ACT_STOP>", None)],
    )
    learned = _trajectory(
        "learned",
        "learned",
        [("<ACT_NEWS>", "News Analyst"), ("<ACT_STOP>", None)],
    )
    learned.provenance["expert_config_hash"] = "different"

    with pytest.raises(ValueError, match="provenance mismatch.*expert_config_hash"):
        evaluate_sets([static], grpo=[learned])
