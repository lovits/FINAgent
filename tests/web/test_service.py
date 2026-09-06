from datetime import date
from pathlib import Path
from time import monotonic

from tradingagents.web.schemas import CreateRunRequest
from tradingagents.web.service import GraphEventProjector, RunManager, RunRecord


def _request(**overrides) -> CreateRunRequest:
    return CreateRunRequest(
        ticker="NVDA",
        analysis_date=date(2026, 8, 20),
        analysts=["market"],
        **overrides,
    )


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
        "node.completed",
        "report.updated",
        "node.completed",
        "report.updated",
        "report.updated",
        "run.completed",
    ]


def test_projector_hides_message_cleanup_nodes() -> None:
    record = RunRecord("run", _request())
    projector = GraphEventProjector(record)
    projector("updates", {"Msg Clear Market": {}, "tools_market": {}})
    assert [event["data"]["node"] for event in record.events] == ["tools_market"]
