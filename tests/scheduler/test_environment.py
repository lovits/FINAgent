from types import SimpleNamespace

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import CallableSchedulerPolicy
from tradingagents.scheduler.trajectory import SchedulerTrajectory, TrajectoryStep
from training.scheduler.environment import TradingAgentsRolloutEnvironment


class _FakeGraph:
    calls = []

    def __init__(self, selected_analysts, *, config, debug, scheduler_policy=None, **kwargs):
        self.config = config
        self.scheduler_policy = scheduler_policy
        self.trajectory_recorder = None
        self.calls.append(config["scheduler_mode"])

    def propagate(self, ticker, trade_date, *, asset_type):
        state = {
            "investment_plan": "plan",
            "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**",
            "final_trade_decision": "**Rating**: Buy",
        }
        if self.config["scheduler_mode"] == "learned":
            trajectory = SchedulerTrajectory(
                trajectory_id="trajectory",
                task_id="placeholder",
                run_id="run",
                mode="learned",
                policy_id=self.scheduler_policy.policy_id,
                ticker=ticker,
                trade_date=trade_date,
                status="accepted",
            )
            trajectory.steps.append(
                TrajectoryStep(
                    step_id=0,
                    serialized_state="{}",
                    valid_actions=[SchedulerAction.MARKET.value],
                    selected_action=SchedulerAction.MARKET.value,
                    policy_id=self.scheduler_policy.policy_id,
                    logprob=-0.1,
                )
            )
            self.trajectory_recorder = SimpleNamespace(current=trajectory)
        return state, "BUY"


def test_environment_runs_learned_policy_and_caches_static_reference():
    _FakeGraph.calls = []
    environment = TradingAgentsRolloutEnvironment(
        {"results_dir": "/tmp"},
        selected_analysts=("market",),
        graph_factory=_FakeGraph,
    )
    policy = CallableSchedulerPolicy(lambda context: SchedulerAction.MARKET)
    task = {"task_id": "task-1", "ticker": "NVDA", "trade_date": "2025-01-02"}

    first = environment.run(task, policy, seed=1)
    second = environment.run(task, policy, seed=2)

    assert first.trajectory.task_id == "task-1"
    assert first.static_state["final_trade_decision"] == "**Rating**: Buy"
    assert second.static_state == first.static_state
    assert _FakeGraph.calls.count("static") == 1
    assert _FakeGraph.calls.count("learned") == 2
