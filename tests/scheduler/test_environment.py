from copy import deepcopy

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import PolicyDecision, SchedulerContext
from tradingagents.scheduler.policy import CallableSchedulerPolicy
from training.scheduler.environment import TradingAgentsSchedulerEnvironment


def _initial_state() -> dict:
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2026-01-05",
        "asset_type": "stock",
        "instrument_context": "Apple Inc.",
        "past_context": "",
        "messages": [],
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {"history": "", "count": 0},
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {"history": "", "count": 0},
        "final_trade_decision": "",
    }


class _Propagator:
    @staticmethod
    def get_graph_args(callbacks=None) -> dict:
        return {}


class _StaticEventGraph:
    @staticmethod
    def stream(initial_state, **kwargs):
        state = deepcopy(initial_state)
        updates = (
            ("Market Analyst", {"market_report": "market evidence"}),
            ("Msg Clear Market", {}),
            (
                "Bull Researcher",
                {"investment_debate_state": {"history": "Bull", "count": 1}},
            ),
            (
                "Bear Researcher",
                {"investment_debate_state": {"history": "Bull\nBear", "count": 2}},
            ),
            ("Research Manager", {"investment_plan": "Hold"}),
            ("Trader", {"trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD"}),
            (
                "Aggressive Analyst",
                {"risk_debate_state": {"history": "Aggressive", "count": 1}},
            ),
            (
                "Conservative Analyst",
                {"risk_debate_state": {"history": "Aggressive\nConservative", "count": 2}},
            ),
            (
                "Neutral Analyst",
                {
                    "risk_debate_state": {
                        "history": "Aggressive\nConservative\nNeutral",
                        "count": 3,
                    }
                },
            ),
            ("Portfolio Manager", {"final_trade_decision": "**Rating**: Hold"}),
        )
        for node_name, update in updates:
            state.update(deepcopy(update))
            yield "updates", {node_name: deepcopy(update)}
            yield "values", deepcopy(state)


class _FakeTradingGraph:
    def __init__(self, **kwargs):
        self.graph = _StaticEventGraph()
        self.propagator = _Propagator()

    @staticmethod
    def create_initial_state(*args, **kwargs) -> dict:
        return _initial_state()


def test_static_environment_projects_real_nodes_into_scheduler_steps() -> None:
    environment = TradingAgentsSchedulerEnvironment(
        {
            "scheduler_max_steps": 16,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
        },
        selected_analysts=("market",),
        graph_factory=_FakeTradingGraph,
    )
    result = environment.run(
        {
            "task_id": "AAPL_2026-01-05_quiet_control",
            "ticker": "AAPL",
            "trade_date": "2026-01-05",
            "asset_type": "stock",
            "data_snapshot_id": "snapshot-1",
            "dataset_version": "tasks-v1",
        },
        mode="static",
        run_id="run-1",
        trajectory_id="trajectory-1",
    )

    trajectory = result.trajectory
    assert trajectory.execution_status == "completed"
    assert trajectory.steps[0].selected_action == "<ACT_MARKET>"
    assert trajectory.steps[-1].selected_action == "<ACT_STOP>"
    assert trajectory.steps[0].state_after["market_report"] == "market evidence"
    assert trajectory.final_outputs["final_trade_decision"] == "**Rating**: Hold"
    assert len(trajectory.node_executions) == 10
    assert trajectory.provenance["task_dataset_version"] == "tasks-v1"
    assert trajectory.provenance["data_snapshot_id"] == "snapshot-1"
    assert trajectory.provenance["action_schema_version"] == "scheduler-actions-v1"
    assert trajectory.provenance["teacher_model"] is None
    assert len(trajectory.provenance["generation_config_hash"]) == 64


class _QueuedDecisionGraph:
    def __init__(self, on_decision):
        self.on_decision = on_decision

    def _queue(self, state, action, valid, step):
        context = SchedulerContext(
            "task-queued",
            deepcopy(state),
            f"prompt-{step}",
            valid,
            ("market", "news"),
            step=step,
        )
        self.on_decision(
            context,
            PolicyDecision(action, policy_id="queued-policy"),
        )

    def stream(self, initial_state, **kwargs):
        state = deepcopy(initial_state)
        self._queue(state, SchedulerAction.MARKET, (SchedulerAction.MARKET,), 0)
        yield "updates", {"Scheduler": {"scheduler_action": "<ACT_MARKET>"}}
        yield "values", deepcopy(state)

        state["market_report"] = "market"
        yield "updates", {"Market Analyst": {"market_report": "market"}}
        yield "values", deepcopy(state)
        yield "updates", {"Msg Clear Market": {}}
        self._queue(state, SchedulerAction.NEWS, (SchedulerAction.NEWS,), 1)
        yield "values", deepcopy(state)

        yield "updates", {"Scheduler": {"scheduler_action": "<ACT_NEWS>"}}
        yield "values", deepcopy(state)
        state["news_report"] = "news"
        state["investment_plan"] = "Hold"
        state["trader_investment_plan"] = "FINAL TRANSACTION PROPOSAL: HOLD"
        state["final_trade_decision"] = "**Rating**: Hold"
        yield "updates", {"News Analyst": {"news_report": "news"}}
        yield "values", deepcopy(state)
        yield "updates", {"Msg Clear News": {}}
        self._queue(state, SchedulerAction.STOP, (SchedulerAction.STOP,), 2)
        yield "values", deepcopy(state)

        yield "updates", {"Scheduler": {"scheduler_action": "<ACT_STOP>"}}
        yield "values", deepcopy(state)


class _QueuedGraphFactory:
    def __init__(self, *, scheduler_on_decision, **kwargs):
        self.graph = _QueuedDecisionGraph(scheduler_on_decision)
        self.propagator = _Propagator()

    @staticmethod
    def create_initial_state(*args, **kwargs):
        return _initial_state()


def test_dynamic_capture_queues_eager_next_decision_until_observation() -> None:
    environment = TradingAgentsSchedulerEnvironment(
        {},
        selected_analysts=("market", "news"),
        graph_factory=_QueuedGraphFactory,
    )
    result = environment.run(
        {
            "task_id": "task-queued",
            "ticker": "AAPL",
            "trade_date": "2026-01-05",
            "data_snapshot_id": "snapshot-1",
        },
        mode="learned",
        run_id="run-queued",
        policy=CallableSchedulerPolicy(lambda _: SchedulerAction.MARKET),
    )

    assert result.trajectory.execution_status == "completed"
    assert [step.selected_action for step in result.trajectory.steps] == [
        "<ACT_MARKET>",
        "<ACT_NEWS>",
        "<ACT_STOP>",
    ]
