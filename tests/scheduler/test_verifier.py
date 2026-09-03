from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.trajectory import SchedulerTrajectory, TrajectoryStep
from tradingagents.scheduler.verifier import VerificationLimits, verify_trajectory


def _trajectory(actions):
    trajectory = SchedulerTrajectory(
        trajectory_id="traj",
        task_id="task",
        run_id="run",
        mode="learned",
        policy_id="policy",
        ticker="NVDA",
        trade_date="2025-01-02",
    )
    for index, action in enumerate(actions):
        trajectory.add_step(
            TrajectoryStep(
                step_id=index,
                serialized_state="{}",
                valid_actions=[action.value],
                selected_action=action.value,
                policy_id="policy",
                agent_node=None if action is SchedulerAction.STOP else "Agent",
            )
        )
    return trajectory


def test_verifier_accepts_complete_bounded_trajectory():
    result = verify_trajectory(
        _trajectory([SchedulerAction.MARKET, SchedulerAction.TRADER, SchedulerAction.STOP]),
        {
            "investment_plan": "plan",
            "trader_investment_plan": "Buy",
            "final_trade_decision": "Overweight",
        },
    )

    assert result.accepted is True
    assert result.reason is None


def test_verifier_rejects_incomplete_loop_and_budget():
    incomplete = verify_trajectory(_trajectory([SchedulerAction.MARKET]), {})
    assert incomplete.reason == "incomplete"

    loop = verify_trajectory(
        _trajectory([SchedulerAction.MARKET] * 4),
        {
            "investment_plan": "plan",
            "trader_investment_plan": "Buy",
            "final_trade_decision": "Overweight",
        },
    )
    assert loop.reason == "loop"

    budget = verify_trajectory(
        _trajectory([SchedulerAction.MARKET, SchedulerAction.NEWS]),
        {
            "investment_plan": "plan",
            "trader_investment_plan": "Buy",
            "final_trade_decision": "Overweight",
        },
        limits=VerificationLimits(max_steps=1),
    )
    assert budget.reason == "budget"
