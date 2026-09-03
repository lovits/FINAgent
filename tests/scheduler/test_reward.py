import pytest

from tradingagents.scheduler.trajectory import (
    ExecutionCost,
    SchedulerStep,
    SchedulerTrajectory,
)
from training.scheduler.reward import portfolio_similarity, score_trajectory


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
    )
    final_state = {"final_trade_decision": f"**Rating**: {rating}"}
    trajectory.add_step(
        SchedulerStep(
            0,
            {},
            "prompt",
            ["<ACT_NEWS>"],
            "<ACT_NEWS>",
            "News Analyst",
            state_after={"news_report": "evidence"},
        )
    )
    trajectory.add_step(
        SchedulerStep(
            1,
            final_state,
            "stop",
            ["<ACT_STOP>"],
            "<ACT_STOP>",
            None,
            state_after=final_state,
        )
    )
    trajectory.final_outputs = {
        "investment_plan": "Hold",
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
        "final_trade_decision": f"**Rating**: {rating}",
    }
    trajectory.cost_total = ExecutionCost(agent_calls=10, tool_calls=4)
    trajectory.execution_status = "completed"
    return trajectory


def test_reward_preserves_quality_and_subtracts_measured_cost() -> None:
    static = _trajectory("static", "static")
    learned = _trajectory("learned", "learned")
    reward = score_trajectory(learned, static)

    assert reward.portfolio_quality == pytest.approx(0.7)
    assert reward.trader_quality == pytest.approx(0.3)
    assert reward.agent_cost == pytest.approx(0.2)
    assert reward.tool_cost == pytest.approx(0.02)
    assert reward.total == pytest.approx(1.08)


def test_fallback_does_not_receive_quality_or_completion_reward() -> None:
    static = _trajectory("static", "static")
    learned = _trajectory("learned", "learned")
    learned.execution_status = "fallback"
    reward = score_trajectory(learned, static)
    assert reward.portfolio_quality == 0
    assert reward.completion == 0
    assert reward.fallback == 0.25


def test_portfolio_similarity_uses_five_tier_distance() -> None:
    assert portfolio_similarity("**Rating**: Buy", "**Rating**: Sell") == 0
    assert portfolio_similarity("**Rating**: Hold", "**Rating**: Overweight") == 0.75
