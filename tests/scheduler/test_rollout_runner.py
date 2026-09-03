from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import CallableSchedulerPolicy
from tradingagents.scheduler.trajectory import SchedulerTrajectory, TrajectoryStep
from training.scheduler.rollout_runner import GroupRolloutRunner, RolloutResult, grpo_rows


class _Environment:
    def run(self, task, policy, *, seed):
        agent_calls = seed + 1
        trajectory = SchedulerTrajectory(
            trajectory_id=f"trajectory-{seed}",
            task_id=task["task_id"],
            run_id=f"run-{seed}",
            mode="learned",
            policy_id=policy.policy_id,
            ticker="NVDA",
            trade_date="2025-01-02",
        )
        for index in range(agent_calls):
            trajectory.add_step(
                TrajectoryStep(
                    step_id=index,
                    serialized_state=f'{{"step":{index}}}',
                    valid_actions=[SchedulerAction.MARKET.value],
                    selected_action=SchedulerAction.MARKET.value,
                    policy_id=policy.policy_id,
                    agent_node="Market Analyst",
                    logprob=-0.2,
                    metadata={"ref_logprob": -0.3},
                )
            )
        state = {
            "investment_plan": "plan",
            "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**",
            "final_trade_decision": "**Rating**: Buy",
        }
        return RolloutResult(trajectory, state, state)


def test_group_rollout_scores_same_task_and_builds_training_rows():
    policy = CallableSchedulerPolicy(lambda context: SchedulerAction.MARKET)
    scored = GroupRolloutRunner(_Environment(), group_size=4).run_task(
        {"task_id": "task"}, policy, base_seed=0
    )

    assert len(scored) == 4
    assert scored[0].reward.total > scored[-1].reward.total
    assert scored[0].advantage > scored[-1].advantage
    rows = grpo_rows(scored)
    assert len(rows) == 1 + 2 + 3 + 4
    assert all(row["task_id"] == "task" for row in rows)
    assert all("reward_components" in row for row in rows)
