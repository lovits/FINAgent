from types import SimpleNamespace

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.scheduler.scheduler_node import SchedulerFallbackError


class _FailingGraph:
    def stream(self, initial_state, **args):
        yield {"scheduler_action": "<ACT_MARKET>"}
        raise SchedulerFallbackError("invalid_policy_decision")


class _StaticGraph:
    def stream(self, initial_state, **args):
        yield {"final_trade_decision": "**Rating**: Hold"}


class _StaticWorkflow:
    def compile(self):
        return _StaticGraph()


def test_static_is_default_and_secret_is_not_configuration():
    assert DEFAULT_CONFIG["scheduler_mode"] == "static"
    assert DEFAULT_CONFIG["teacher_model"] == "google/gemini-3.8-flash"
    assert "openrouter_api_key" not in DEFAULT_CONFIG


def test_stream_restarts_original_task_with_static_graph_on_scheduler_failure():
    graph = object.__new__(TradingAgentsGraph)
    graph.graph = _FailingGraph()
    graph.static_workflow = _StaticWorkflow()
    graph.scheduler_mode = "learned"
    graph.scheduler_fallback_reason = None
    graph.config = {"scheduler_fallback_enabled": True}
    graph.propagator = SimpleNamespace(
        create_initial_state=lambda *args, **kwargs: {"fresh": True}
    )
    initial = {
        "company_of_interest": "NVDA",
        "trade_date": "2025-01-02",
        "asset_type": "stock",
        "past_context": "",
        "instrument_context": "NVIDIA",
    }

    chunks = list(graph.stream_graph(initial))

    assert chunks[0]["scheduler_action"] == "<ACT_MARKET>"
    assert chunks[1] == {
        "scheduler_requested_mode": "learned",
        "scheduler_mode": "static",
        "scheduler_fallback_reason": "invalid_policy_decision",
    }
    assert chunks[2]["final_trade_decision"] == "**Rating**: Hold"
    assert graph.scheduler_fallback_reason == "invalid_policy_decision"


def test_stream_reports_startup_fallback_before_static_chunks():
    graph = object.__new__(TradingAgentsGraph)
    graph.graph = _StaticGraph()
    graph.requested_scheduler_mode = "learned"
    graph.scheduler_mode = "static"
    graph.scheduler_fallback_reason = "scheduler_policy_unavailable"
    graph.config = {"scheduler_fallback_enabled": True}

    chunks = list(graph.stream_graph({}))

    assert chunks[0]["scheduler_fallback_reason"] == "scheduler_policy_unavailable"
    assert chunks[1]["final_trade_decision"] == "**Rating**: Hold"
