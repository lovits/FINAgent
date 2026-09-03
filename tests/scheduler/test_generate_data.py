import json

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.trajectory import SchedulerTrajectory, TrajectoryStep
from training.scheduler import generate_data
from training.scheduler.rollout_runner import RolloutResult


class _Environment:
    def __init__(self, config, *, selected_analysts):
        self.config = config

    def run(self, task, policy, *, seed):
        trajectory = SchedulerTrajectory(
            trajectory_id=f"trajectory-{seed}",
            task_id=task["task_id"],
            run_id=f"run-{seed}",
            mode="learned",
            policy_id=policy.policy_id,
            ticker=task["ticker"],
            trade_date=task["trade_date"],
        )
        trajectory.add_step(
            TrajectoryStep(
                step_id=0,
                serialized_state="{}",
                valid_actions=[SchedulerAction.MARKET.value],
                selected_action=SchedulerAction.MARKET.value,
                policy_id=policy.policy_id,
                agent_node="Market Analyst",
            )
        )
        state = {
            "investment_plan": "plan",
            "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**",
            "final_trade_decision": "**Rating**: Buy",
        }
        return RolloutResult(trajectory, state, state)


def test_load_tasks_and_generate_static_data_without_teacher_key(tmp_path, monkeypatch):
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(
        json.dumps({"ticker": "NVDA", "trade_date": "2025-01-02"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(generate_data, "TradingAgentsRolloutEnvironment", _Environment)

    tasks = generate_data.load_tasks(task_path)
    counts = generate_data.generate(
        tasks=tasks,
        output_dir=tmp_path / "output",
        policy_kind="static",
        trajectories_per_task=1,
        selected_analysts=("market",),
        min_rating_similarity=0.5,
    )

    assert counts == {"accepted": 1, "rejected": 0}
    saved = (tmp_path / "output" / "accepted.jsonl").read_text(encoding="utf-8")
    assert '"source":"static_teacher"' in saved
    assert "OPENROUTER_API_KEY" not in saved
