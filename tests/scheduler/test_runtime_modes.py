from pathlib import Path

import pytest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import CallableSchedulerPolicy


class _DummyLLM:
    def with_structured_output(self, *args, **kwargs):
        return self

    def bind_tools(self, *args, **kwargs):
        return self


class _DummyClient:
    def get_llm(self):
        return _DummyLLM()


def _config(tmp_path: Path, mode: str) -> dict:
    return {
        **DEFAULT_CONFIG,
        "orchestration_mode": mode,
        "results_dir": str(tmp_path / "results"),
        "data_cache_dir": str(tmp_path / "cache"),
        "memory_log_path": None,
    }


def test_static_mode_builds_original_graph_without_policy(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.create_llm_client", lambda **_: _DummyClient()
    )
    graph = TradingAgentsGraph(selected_analysts=("market",), config=_config(tmp_path, "static"))
    nodes = graph.graph.get_graph().nodes

    assert graph.orchestration_mode == "static"
    assert "Scheduler" not in nodes


@pytest.mark.parametrize("mode", ["teacher", "learned"])
def test_dynamic_modes_build_scheduler_graph(monkeypatch, tmp_path, mode: str) -> None:
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.create_llm_client", lambda **_: _DummyClient()
    )
    policy = CallableSchedulerPolicy(
        lambda _: SchedulerAction.MARKET,
        policy_id=f"{mode}-test",
    )
    graph = TradingAgentsGraph(
        selected_analysts=("market",),
        config=_config(tmp_path, mode),
        scheduler_policy=policy,
    )

    assert graph.orchestration_mode == mode
    assert "Scheduler" in graph.graph.get_graph().nodes


def test_dynamic_mode_requires_policy(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires a scheduler_policy"):
        TradingAgentsGraph(config=_config(tmp_path, "teacher"))


def test_unknown_mode_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="orchestration_mode"):
        TradingAgentsGraph(config=_config(tmp_path, "random"))


def test_checkpoint_signature_includes_orchestration_mode() -> None:
    graph = object.__new__(TradingAgentsGraph)
    graph.selected_analysts = ("market",)
    graph.orchestration_mode = "teacher"
    graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}

    assert "orchestration=teacher" in graph._run_signature("stock")
