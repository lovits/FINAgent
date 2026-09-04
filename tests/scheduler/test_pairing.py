from tradingagents.scheduler.trajectory import ExecutionCost, SchedulerTrajectory
from training.scheduler.pair import pair_trajectories


def _trajectory(identifier: str, mode: str, rating: str = "Hold") -> SchedulerTrajectory:
    trajectory = SchedulerTrajectory(
        identifier,
        "run-1",
        "task-1",
        mode,  # type: ignore[arg-type]
        f"{mode}-v1",
        "AAPL",
        "2026-01-05",
        data_snapshot_id="snapshot-1",
        execution_status="completed",
        audit_status="accepted",
    )
    trajectory.final_outputs = {
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
        "final_trade_decision": f"**Rating**: {rating}",
    }
    trajectory.cost_total = ExecutionCost(agent_calls=10, tool_calls=4)
    trajectory.provenance = {
        "task_dataset_version": "scheduler-tasks-v2",
        "information_cutoff": "2026-01-05T23:59:59Z",
        "selected_analysts": ["market", "news"],
        "research_depth": "shallow",
        "output_language": "Chinese",
        "expert_config_hash": "expert-config-1",
        "scheduler_max_steps": 16,
    }
    return trajectory


def test_pair_accepts_close_decision_and_cost() -> None:
    static = _trajectory("static-1", "static")
    teacher = _trajectory("teacher-1", "teacher", rating="Overweight")
    comparison = pair_trajectories(static, teacher)

    assert comparison.pair_status == "teacher_verified"
    assert comparison.portfolio_rating_distance == 1
    assert comparison.teacher_to_static_cost_ratio == 1.0


def test_pair_rejects_distant_rating_and_high_cost() -> None:
    static = _trajectory("static-1", "static", rating="Buy")
    teacher = _trajectory("teacher-1", "teacher", rating="Sell")
    teacher.cost_total = ExecutionCost(agent_calls=20, tool_calls=8)
    comparison = pair_trajectories(static, teacher)

    assert comparison.pair_status == "rejected"
    assert "portfolio_rating_too_far" in comparison.rejection_reasons
    assert "teacher_cost_too_high" in comparison.rejection_reasons


def test_pair_rejects_mismatched_expert_configuration() -> None:
    static = _trajectory("static-1", "static")
    teacher = _trajectory("teacher-1", "teacher")
    teacher.provenance["expert_config_hash"] = "different"

    comparison = pair_trajectories(static, teacher)

    assert comparison.pair_status == "rejected"
    assert "provenance_mismatch:expert_config_hash" in comparison.rejection_reasons
