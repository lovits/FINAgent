from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.trajectory import SchedulerTrajectory, TrajectoryStep
from training.scheduler.reward import parse_portfolio_rating, score_trajectory

STATIC_STATE = {
    "investment_plan": "plan",
    "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**",
    "final_trade_decision": "**Rating**: Buy",
}


def _trajectory(agent_calls, *, complete=True):
    trajectory = SchedulerTrajectory(
        trajectory_id=f"traj-{agent_calls}-{complete}",
        task_id="task",
        run_id="run",
        mode="learned",
        policy_id="policy",
        ticker="NVDA",
        trade_date="2025-01-02",
    )
    actions = [SchedulerAction.MARKET] * agent_calls
    actions.append(SchedulerAction.STOP)
    for index, action in enumerate(actions):
        trajectory.add_step(
            TrajectoryStep(
                step_id=index,
                serialized_state="{}",
                valid_actions=[action.value],
                selected_action=action.value,
                policy_id="policy",
                agent_node=None if action is SchedulerAction.STOP else "Agent",
                tool_calls=1 if action is not SchedulerAction.STOP else 0,
                input_tokens=100,
                output_tokens=100,
            )
        )
    state = dict(STATIC_STATE) if complete else {}
    return trajectory, state


def test_reward_prefers_complete_low_cost_over_complete_high_cost_and_incomplete():
    low_cost, low_state = _trajectory(4)
    high_cost, high_state = _trajectory(10)
    incomplete, incomplete_state = _trajectory(1, complete=False)

    reward_a = score_trajectory(low_cost, low_state, STATIC_STATE).total
    reward_b = score_trajectory(high_cost, high_state, STATIC_STATE).total
    reward_c = score_trajectory(incomplete, incomplete_state, STATIC_STATE).total

    assert reward_a > reward_b > reward_c


def test_reward_reduces_portfolio_quality_by_rating_distance():
    trajectory, state = _trajectory(4)
    matching = score_trajectory(trajectory, state, STATIC_STATE)
    state["final_trade_decision"] = "**Rating**: Sell"
    opposite = score_trajectory(trajectory, state, STATIC_STATE)

    assert matching.portfolio_quality == 0.7
    assert matching.format_compliance == 0.1
    assert opposite.portfolio_quality == 0.0
    assert matching.total > opposite.total


def test_reward_penalizes_error_and_fallback():
    trajectory, state = _trajectory(4)
    trajectory.steps[0].error = "invalid action"
    trajectory.fallback_reason = "invalid_policy_decision"

    reward = score_trajectory(trajectory, state, STATIC_STATE)

    assert reward.invalid == 1.0
    assert reward.fallback == 0.25
    assert reward.portfolio_quality == 0.0
    assert reward.trader_quality == 0.0
    assert reward.completion == 0.0


def test_explicit_rating_wins_and_unparseable_output_is_incomplete():
    assert parse_portfolio_rating("Buy case rejected. **Rating**: Sell") == "Sell"
    trajectory, state = _trajectory(4)
    state["final_trade_decision"] = "I cannot provide a rating."

    reward = score_trajectory(trajectory, state, STATIC_STATE)

    assert reward.portfolio_quality == 0.0
    assert reward.completion == 0.0
    assert reward.incomplete == 1.0


def test_format_reward_requires_exactly_one_terminal_stop():
    trajectory, state = _trajectory(2)
    trajectory.steps.insert(
        1,
        TrajectoryStep(
            step_id=1,
            serialized_state="{}",
            valid_actions=[SchedulerAction.STOP.value],
            selected_action=SchedulerAction.STOP.value,
            policy_id="policy",
        ),
    )

    reward = score_trajectory(trajectory, state, STATIC_STATE)

    assert reward.format_compliance == 0.0
