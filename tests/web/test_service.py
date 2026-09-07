from datetime import date
from pathlib import Path
from time import monotonic

import pytest

from tradingagents.scheduler.cost_tracker import CostSnapshot
from tradingagents.web.schemas import CreateRunRequest
from tradingagents.web.service import (
    GraphEventProjector,
    RunCancelled,
    RunManager,
    RunRecord,
    _report_sections,
    _runtime_config,
)


def _request(**overrides) -> CreateRunRequest:
    values = {
        "ticker": "NVDA",
        "analysis_date": date(2026, 8, 20),
        "analysts": ["market"],
        **overrides,
    }
    return CreateRunRequest(**values)


class FakeGraph:
    def __init__(self, analysts, *, config, debug, callbacks):
        self.config = config

    def propagate(self, ticker, trade_date, *, asset_type, on_graph_event):
        on_graph_event("updates", {"Market Analyst": {"market_report": "Market"}})
        on_graph_event("values", {"market_report": "# Market\nEvidence"})
        on_graph_event("updates", {"Trader": {"trader_investment_plan": "Hold"}})
        final_state = {
            "market_report": "# Market\nEvidence",
            "trader_investment_plan": "Action: HOLD",
            "final_trade_decision": "Rating: Hold",
        }
        on_graph_event("values", final_state)
        return final_state, "HOLD"

    def save_reports(self, final_state, ticker, destination):
        path = Path(destination) / "complete_report.md"
        path.parent.mkdir(parents=True)
        path.write_text("# Complete report", encoding="utf-8")
        return path


def test_run_manager_streams_nodes_reports_and_completion(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tradingagents.web.service.DEFAULT_CONFIG",
        {
            "results_dir": str(tmp_path),
            "scheduler_adapter_path": None,
        },
    )
    manager = RunManager(graph_factory=FakeGraph)
    record = manager.create(_request())
    deadline = monotonic() + 2
    while record.status not in {"completed", "failed"} and monotonic() < deadline:
        record.wait_after(len(record.events) - 1, timeout=0.05)

    assert record.status == "completed"
    assert record.signal == "HOLD"
    assert record.complete_report == "# Complete report"
    assert record.report_sections["market_report"].startswith("# Market")
    assert [event["type"] for event in record.events] == [
        "run.started",
        "node.progress",
        "report.updated",
        "node.progress",
        "report.updated",
        "report.updated",
        "run.completed",
    ]


def test_projector_hides_message_cleanup_nodes() -> None:
    record = RunRecord("run", _request())
    projector = GraphEventProjector(record)
    projector("updates", {"Msg Clear Market": {}, "tools_market": {}})
    assert [event["data"]["node"] for event in record.events] == ["tools_market"]


def test_projector_keeps_cli_message_and_compacts_tool_request() -> None:
    record = RunRecord("run", _request())
    projector = GraphEventProjector(record)
    projector(
        "updates",
        {
            "Market Analyst": {
                "messages": [
                    {
                        "content": "Checking price data",
                        "tool_calls": [{"name": "get_stock_data", "args": {"ticker": "NVDA"}}],
                    }
                ]
            }
        },
    )
    data = record.events[0]["data"]
    assert data["message"] == "Checking price data"
    assert data["tool_calls"] == [
        {"name": "get_stock_data", "argument_keys": ["ticker"]}
    ]


def test_projector_keeps_all_tool_results_and_live_usage() -> None:
    class CostTracker:
        def snapshot(self) -> CostSnapshot:
            return CostSnapshot(
                llm_calls=1,
                tool_calls=2,
                input_tokens=120,
                output_tokens=30,
            )

    record = RunRecord("run", _request())
    projector = GraphEventProjector(record, CostTracker())
    projector(
        "updates",
        {
            "tools_market": {
                "messages": [
                    {
                        "type": "tool",
                        "name": "get_stock_data",
                        "tool_call_id": "call-1",
                        "content": "# Price rows\n# Source: BaoStock\n# Total records: 250\nrow1\nrow2",
                    },
                    {
                        "type": "tool",
                        "name": "get_indicators",
                        "tool_call_id": "call-2",
                        "content": "indicator rows",
                    },
                ]
            }
        },
    )

    data = record.events[0]["data"]
    assert [message["name"] for message in data["messages"]] == [
        "get_stock_data",
        "get_indicators",
    ]
    assert data["messages"][0]["source"] == "BaoStock"
    assert data["messages"][0]["summarized"] is True
    assert "row1" not in data["messages"][0]["content"]
    assert data["usage"]["input_tokens"] == 120
    assert data["cumulative_metrics"]["output_tokens"] == 30
    assert record.metrics["tool_calls"] == 2


def test_projector_adds_scheduler_usage_to_expert_totals() -> None:
    class CostTracker:
        def snapshot(self) -> CostSnapshot:
            return CostSnapshot(llm_calls=2, input_tokens=200, output_tokens=50)

    record = RunRecord("run", _request())
    projector = GraphEventProjector(record, CostTracker())
    projector(
        "updates",
        {
            "Scheduler": {
                "scheduler_action": "<ACT_MARKET>",
                "scheduler_valid_actions": ["<ACT_MARKET>"],
                "scheduler_decision_metadata": {
                    "usage": {
                        "llm_calls": 1,
                        "input_tokens": 40,
                        "output_tokens": 1,
                    }
                },
            }
        },
    )

    data = record.events[0]["data"]
    assert data["usage"]["input_tokens"] == 240
    assert data["cumulative_metrics"] == {
        "llm_calls": 3,
        "tool_calls": 0,
        "input_tokens": 240,
        "output_tokens": 51,
    }


def test_cancel_stops_at_next_graph_event() -> None:
    record = RunRecord("run", _request(), status="running")
    manager = RunManager(graph_factory=FakeGraph)
    manager._runs[record.run_id] = record
    manager.cancel(record.run_id)
    with pytest.raises(RunCancelled):
        GraphEventProjector(record)("values", {})
    assert record.status == "cancelling"


def test_partial_debates_are_exposed_like_cli_reports() -> None:
    reports = _report_sections(
        {
            "investment_debate_state": {"bull_history": "Bull case", "bear_history": "Bear case"},
            "risk_debate_state": {"neutral_history": "Neutral risk"},
        }
    )
    assert "Bull case" in reports["investment_plan"]
    assert "Bear case" in reports["investment_plan"]
    assert "Neutral risk" in reports["final_trade_decision"]


def test_web_retry_always_starts_without_langgraph_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "tradingagents.web.service.DEFAULT_CONFIG",
        {
            "checkpoint_enabled": True,
            "scheduler_adapter_path": None,
        },
    )
    config, *_ = _runtime_config(_request())
    assert config["checkpoint_enabled"] is False


def test_web_english_alias_resolves_to_mainland_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "tradingagents.web.service.DEFAULT_CONFIG",
        {"results_dir": str(tmp_path), "scheduler_adapter_path": None},
    )
    manager = RunManager(graph_factory=FakeGraph)
    record = manager.create(_request(ticker="MOUTAI"))
    deadline = monotonic() + 2
    while record.status not in {"completed", "failed"} and monotonic() < deadline:
        record.wait_after(len(record.events) - 1, timeout=0.05)

    snapshot = record.snapshot()
    assert snapshot["resolved_ticker"] == "600519.SS"
    assert snapshot["data_sources"] == {
        "market": "BaoStock",
        "fundamentals": "BaoStock",
        "news": "AKShare / Eastmoney",
    }


def test_failed_run_exposes_node_type_message_and_recovery(tmp_path, monkeypatch) -> None:
    class FailingGraph(FakeGraph):
        def propagate(self, ticker, trade_date, *, asset_type, on_graph_event):
            on_graph_event("updates", {"Market Analyst": {}})
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "tradingagents.web.service.DEFAULT_CONFIG",
        {"results_dir": str(tmp_path), "scheduler_adapter_path": None},
    )
    manager = RunManager(graph_factory=FailingGraph)
    record = manager.create(_request(ticker="MOUTAI"))
    deadline = monotonic() + 2
    while record.status not in {"completed", "failed"} and monotonic() < deadline:
        record.wait_after(len(record.events) - 1, timeout=0.05)

    assert record.status == "failed"
    assert record.error_details["node"] == "Market Analyst"
    assert record.error_details["type"] == "RuntimeError"
    assert record.error_details["message"] == "provider unavailable"
    assert record.error_details["suggestion"]
