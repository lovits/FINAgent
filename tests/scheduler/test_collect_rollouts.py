import json

import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import PolicyDecision, SchedulerContext
from tradingagents.scheduler.store import TrajectoryStore
from tradingagents.scheduler.trajectory import (
    ExecutionCost,
    SchedulerStep,
    SchedulerTrajectory,
)
from training.scheduler.collect_rollouts import RolloutConfig, collect
from training.scheduler.environment import EnvironmentRunResult


class _ActivePolicy:
    policy_id = "active-sft"

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        return PolicyDecision(context.valid_actions[0], self.policy_id, logprob=0.0)


class _ReferencePolicy(_ActivePolicy):
    policy_id = "reference-sft"

    def action_logprobs(self, context: SchedulerContext):
        return dict.fromkeys(context.valid_actions, 0.0)


class _Environment:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, task, *, mode, run_id, policy, trajectory_id=None):
        index = int(str(trajectory_id).rsplit(":", 1)[-1])
        trajectory = SchedulerTrajectory(
            str(trajectory_id),
            run_id,
            task["task_id"],
            "learned",
            policy.policy_id,
            task["ticker"],
            task["trade_date"],
            data_snapshot_id=task["data_snapshot_id"],
        )
        news_state = {"news_report": ""}
        news_context = SchedulerContext(
            task["task_id"],
            news_state,
            "news prompt",
            (SchedulerAction.NEWS,),
            ("news",),
        )
        news = policy.select_action(news_context)
        trajectory.add_step(
            SchedulerStep(
                0,
                news_state,
                "news prompt",
                ["<ACT_NEWS>"],
                news.action.value,
                "News Analyst",
                state_after={"news_report": "evidence"},
                old_logprob=news.logprob,
                ref_logprob=news.metadata["ref_logprob"],
            )
        )
        ratings = ("Strong Sell", "Sell", "Neutral", "Strong Buy")
        final_state = {
            "investment_plan": "Hold",
            "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
            "final_trade_decision": f"**Rating**: {ratings[index]}",
        }
        stop_context = SchedulerContext(
            task["task_id"],
            final_state,
            "stop prompt",
            (SchedulerAction.STOP,),
            ("news",),
            step=1,
        )
        stop = policy.select_action(stop_context)
        trajectory.add_step(
            SchedulerStep(
                1,
                final_state,
                "stop prompt",
                ["<ACT_STOP>"],
                stop.action.value,
                None,
                state_after=final_state,
                old_logprob=stop.logprob,
                ref_logprob=stop.metadata["ref_logprob"],
            )
        )
        trajectory.final_outputs = dict(final_state)
        trajectory.cost_total = ExecutionCost(agent_calls=index + 1)
        trajectory.execution_status = "completed"
        return EnvironmentRunResult(trajectory, final_state)


def _static_reference() -> SchedulerTrajectory:
    trajectory = SchedulerTrajectory(
        "static-task-1",
        "static-run",
        "task-1",
        "static",
        "static-v1",
        "AAPL",
        "2026-01-05",
        data_snapshot_id="snapshot-1",
        execution_status="completed",
    )
    trajectory.final_outputs = {
        "investment_plan": "Hold",
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
        "final_trade_decision": "**Rating**: Strong Buy",
    }
    trajectory.provenance = {
        "selected_analysts": ["market", "news"],
        "research_depth": "shallow",
        "output_language": "Chinese",
    }
    return trajectory


@pytest.mark.parametrize("reward_mode", ["static_agreement", "automatic"])
def test_collect_rollouts_materializes_four_trajectory_credit_group(
    monkeypatch, tmp_path, reward_mode
) -> None:
    from training.scheduler.auto_review import AutoReward

    class Reviewer:
        def __init__(self, *args):
            self.reviews = {}

        def __call__(self, trajectory, reference):
            score = trajectory.cost_total.agent_calls / 4
            self.reviews[trajectory.trajectory_id] = {"quality": score}
            return AutoReward(score, score, 1, 0)

    monkeypatch.setattr("training.scheduler.collect_rollouts.AutomaticReviewer", Reviewer)
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "ticker": "AAPL",
                "trade_date": "2026-01-05",
                "data_snapshot_id": "snapshot-1",
                "split": "train",
                "selected_analysts": ["market", "news"],
                "research_depth": "shallow",
                "output_language": "Chinese",
            }
        )
        + "\n"
    )
    static_path = tmp_path / "static.jsonl"
    TrajectoryStore(static_path).append(_static_reference())
    monkeypatch.setattr(
        "training.scheduler.collect_rollouts.load_shared_hf_scheduler_policies",
        lambda *args, **kwargs: (_ActivePolicy(), _ReferencePolicy()),
    )
    monkeypatch.setattr(
        "training.scheduler.collect_rollouts.TradingAgentsSchedulerEnvironment",
        _Environment,
    )

    counts = collect(
        RolloutConfig(
            tasks_path=str(tasks_path),
            static_trajectories_path=str(static_path) if reward_mode == "static_agreement" else "not-required.jsonl",
            active_adapter_path="adapter/active",
            reference_adapter_path="adapter/reference",
            output_dir=str(tmp_path / "rollouts"),
            run_id="iteration-1",
            reward_config_path=None,
            base_model="tiny",
            base_revision=None,
            reward_mode=reward_mode,
        )
    )

    expected = {"tasks": 1, "trajectories": 4, "rows": 8}
    if reward_mode == "automatic":
        expected["skipped_groups"] = 0
        assert (tmp_path / "rollouts/quality_reviews.json").is_file()
    assert counts == expected
    rows = [
        json.loads(line)
        for line in (tmp_path / "rollouts" / "grpo.jsonl").read_text().splitlines()
    ]
    advantages = {row["trajectory_id"]: row["advantage"] for row in rows}
    assert len(advantages) == 4
    assert sum(advantages.values()) == pytest.approx(0.0)
    assert min(advantages.values()) < 0 < max(advantages.values())
    manifest = json.loads((tmp_path / "rollouts" / "rollout_manifest.json").read_text())
    assert manifest["config"]["group_size"] == 4
