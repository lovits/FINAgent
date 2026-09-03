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
