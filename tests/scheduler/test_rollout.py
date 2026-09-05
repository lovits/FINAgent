from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import PolicyDecision, SchedulerContext
from tradingagents.scheduler.trajectory import SchedulerStep, SchedulerTrajectory
from training.scheduler.environment import EnvironmentRunResult
from training.scheduler.rollout import GroupRolloutRunner, grpo_rows


class _ActivePolicy:
    policy_id = "active-v1"

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        action = context.valid_actions[0]
        return PolicyDecision(action, self.policy_id, logprob=0.0)


class _ReferencePolicy(_ActivePolicy):
    policy_id = "reference-v1"

    def action_logprobs(self, context: SchedulerContext):
        return dict.fromkeys(context.valid_actions, 0.0)


class _Environment:
    def run(self, task, *, mode, run_id, policy, trajectory_id=None):
        trajectory = SchedulerTrajectory(
            trajectory_id,
            run_id,
            task["task_id"],
            "learned",
            policy.policy_id,
            task["ticker"],
            task["trade_date"],
            data_snapshot_id=task["data_snapshot_id"],
        )
        state = {"news_report": ""}
        news_context = SchedulerContext(
            task["task_id"],
            state,
            "news prompt",
            (SchedulerAction.NEWS,),
            ("news",),
        )
        news = policy.select_action(news_context)
        trajectory.add_step(
            SchedulerStep(
                0,
                state,
                "news prompt",
                ["<ACT_NEWS>"],
                news.action.value,
                "News Analyst",
                state_after={"news_report": "evidence"},
                old_logprob=news.logprob,
                ref_logprob=news.metadata["ref_logprob"],
            )
        )
        final_state = {"final_trade_decision": "**Rating**: Hold"}
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
        trajectory.final_outputs = {
            "investment_plan": "Hold",
            "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
            "final_trade_decision": "**Rating**: Hold",
        }
        trajectory.execution_status = "completed"
        return EnvironmentRunResult(trajectory, final_state)


def _static_reference() -> SchedulerTrajectory:
    trajectory = SchedulerTrajectory(
        "static-1",
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
        "final_trade_decision": "**Rating**: Hold",
    }
    return trajectory


def test_group_rollout_produces_four_scored_trajectories_and_rows() -> None:
    runner = GroupRolloutRunner(_Environment())
    scored = runner.run_task(
        {
            "task_id": "task-1",
            "ticker": "AAPL",
            "trade_date": "2026-01-05",
            "data_snapshot_id": "snapshot-1",
        },
        active_policy=_ActivePolicy(),
        reference_policy=_ReferencePolicy(),
        static_reference=_static_reference(),
        run_id="rollout-1",
        base_seed=42,
    )

    assert len(scored) == 4
    assert [value.advantage for value in scored] == [0.0] * 4
    rows = grpo_rows(scored)
    assert len(rows) == 8
    assert all(row["old_logprob"] == 0 for row in rows)
    assert all(row["ref_logprob"] == 0 for row in rows)


def test_judge_failure_retains_raw_group_and_produces_no_fake_score():
    import pytest

    from training.scheduler.auto_review import ReviewUnavailable

    raw = []

    def unavailable(*args):
        raise ReviewUnavailable("judge unavailable")

    runner = GroupRolloutRunner(_Environment(), reward_scorer=unavailable, trajectory_sink=raw.append)
    with pytest.raises(ReviewUnavailable):
        runner.run_task({"task_id": "task-1", "ticker": "AAPL", "trade_date": "2026-01-05",
                         "data_snapshot_id": "snapshot-1"}, active_policy=_ActivePolicy(),
                        reference_policy=_ReferencePolicy(), static_reference=_static_reference(),
                        run_id="group", base_seed=42)
    assert len(raw) == 4
    assert all(trajectory.reward is None for trajectory in raw)


def test_candidate_execution_and_reward_review_are_parallel():
    from threading import Barrier, Lock

    execution_gate = Barrier(4, timeout=2)
    review_gate = Barrier(4, timeout=2)

    class ParallelEnvironment(_Environment):
        def run(self, *args, **kwargs):
            execution_gate.wait()
            return super().run(*args, **kwargs)

    def reward(trajectory, reference):
        from training.scheduler.auto_review import AutoReward
        review_gate.wait()
        value = int(trajectory.trajectory_id.rsplit(":", 1)[-1]) / 3
        return AutoReward(value, value, 1, 0)

    runner = GroupRolloutRunner(
        ParallelEnvironment(), reward_scorer=reward,
        policy_lock=Lock(), trajectory_workers=4,
    )
    scored = runner.run_task(
        {"task_id": "task-1", "ticker": "AAPL", "trade_date": "2026-01-05",
         "data_snapshot_id": "snapshot-1"},
        active_policy=_ActivePolicy(), reference_policy=_ReferencePolicy(),
        static_reference=None, run_id="parallel", base_seed=42,
    )
    assert len(scored) == 4
    assert min(item.advantage for item in scored) < 0 < max(item.advantage for item in scored)


def test_existing_trajectory_is_reused_without_execution():
    existing = _Environment().run(
        {"task_id": "task-1", "ticker": "AAPL", "trade_date": "2026-01-05",
         "data_snapshot_id": "snapshot-1"}, mode="learned", run_id="resume",
        policy=__import__("tradingagents.scheduler.policy", fromlist=["ReferenceScoredPolicy"]).ReferenceScoredPolicy(
            _ActivePolicy(), _ReferencePolicy()), trajectory_id="resume:task-1:0",
    ).trajectory

    class CountingEnvironment(_Environment):
        calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            return super().run(*args, **kwargs)

    environment = CountingEnvironment()
    runner = GroupRolloutRunner(environment)
    runner.run_task(
        {"task_id": "task-1", "ticker": "AAPL", "trade_date": "2026-01-05",
         "data_snapshot_id": "snapshot-1"}, active_policy=_ActivePolicy(),
        reference_policy=_ReferencePolicy(), static_reference=_static_reference(),
        run_id="resume", base_seed=42, existing_trajectories=[existing],
    )
    assert environment.calls == 3
