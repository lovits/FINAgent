from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any

from tradingagents.dataflows.symbol_utils import normalize_symbol
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.scheduler.cost_tracker import SchedulerCostCallback

from .schemas import CreateRunRequest

ANALYST_ORDER = ("market", "social", "news", "fundamentals")
DEPTH_ROUNDS = {"shallow": 1, "medium": 3, "deep": 5}
CRYPTO_SUFFIXES = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH")

REPORT_FIELDS = {
    "market_report": "Market Analysis",
    "sentiment_report": "Sentiment Analysis",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Decision",
    "trader_investment_plan": "Trading Plan",
    "final_trade_decision": "Portfolio Decision",
}


def _now_ms() -> int:
    return int(time() * 1000)


@dataclass
class RunRecord:
    run_id: str
    request: CreateRunRequest
    status: str = "queued"
    created_at_ms: int = field(default_factory=_now_ms)
    updated_at_ms: int = field(default_factory=_now_ms)
    events: list[dict[str, Any]] = field(default_factory=list)
    report_sections: dict[str, str] = field(default_factory=dict)
    complete_report: str | None = None
    signal: str | None = None
    error: str | None = None
    metrics: dict[str, int] = field(default_factory=dict)
    _condition: threading.Condition = field(
        default_factory=threading.Condition, repr=False
    )

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        with self._condition:
            event = {
                "id": len(self.events),
                "type": event_type,
                "timestamp_ms": _now_ms(),
                "data": data,
            }
            self.events.append(event)
            self.updated_at_ms = event["timestamp_ms"]
            self._condition.notify_all()

    def wait_after(self, cursor: int, timeout: float = 15.0) -> list[dict[str, Any]]:
        with self._condition:
            if len(self.events) <= cursor + 1 and self.status not in {"completed", "failed"}:
                self._condition.wait(timeout)
            return [event for event in self.events if event["id"] > cursor]

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "request": self.request.model_dump(mode="json"),
            "report_sections": dict(self.report_sections),
            "complete_report": self.complete_report,
            "signal": self.signal,
            "error": self.error,
            "metrics": dict(self.metrics),
            "event_count": len(self.events),
        }


class GraphEventProjector:
    def __init__(self, record: RunRecord):
        self.record = record
        self._reports: dict[str, str] = {}

    def __call__(self, stream_mode: str, payload: object) -> None:
        if stream_mode == "updates" and isinstance(payload, dict):
            self._project_nodes(payload)
        elif stream_mode == "values" and isinstance(payload, dict):
            self._project_reports(payload)

    def _project_nodes(self, payload: dict[str, Any]) -> None:
        for node_name, update in payload.items():
            if node_name.startswith("Msg Clear"):
                continue
            data = {"node": node_name, "kind": _node_kind(node_name)}
            if isinstance(update, dict) and update.get("scheduler_action"):
                data["selected_action"] = update["scheduler_action"]
                data["valid_actions"] = update.get("scheduler_valid_actions", [])
            self.record.emit("node.completed", data)

    def _project_reports(self, state: dict[str, Any]) -> None:
        reports = _report_sections(state)
        for key, content in reports.items():
            if not content or self._reports.get(key) == content:
                continue
            self._reports[key] = content
            self.record.report_sections[key] = content
            self.record.emit(
                "report.updated",
                {"section": key, "title": REPORT_FIELDS[key], "content": content},
            )


class RunManager:
    def __init__(
        self,
        graph_factory: Callable[..., TradingAgentsGraph] = TradingAgentsGraph,
    ):
        self.graph_factory = graph_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-run")
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}

    def create(self, request: CreateRunRequest) -> RunRecord:
        with self._lock:
            if any(run.status in {"queued", "running"} for run in self._runs.values()):
                raise RuntimeError("another analysis is already running")
            record = RunRecord(uuid.uuid4().hex, request)
            self._runs = {record.run_id: record}
            self._executor.submit(self._execute, record)
            return record

    def get(self, run_id: str) -> RunRecord:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown run: {run_id}") from exc

    def _execute(self, record: RunRecord) -> None:
        record.status = "running"
        record.emit("run.started", {"run_id": record.run_id})
        try:
            self._run_graph(record)
        except Exception as exc:  # noqa: BLE001 - boundary converts failure to run state
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            record.emit("run.failed", {"error": record.error})

    def _run_graph(self, record: RunRecord) -> None:
        config, ticker, asset_type, analysts = _runtime_config(record.request)
        cost_tracker = SchedulerCostCallback()
        graph = self.graph_factory(
            analysts,
            config=config,
            debug=False,
            callbacks=[cost_tracker],
        )
        final_state, signal = graph.propagate(
            ticker,
            str(record.request.analysis_date),
            asset_type=asset_type,
            on_graph_event=GraphEventProjector(record),
        )
        destination = Path(config["results_dir"]) / "web" / record.run_id
        report_path = graph.save_reports(final_state, ticker, destination)
        record.complete_report = report_path.read_text(encoding="utf-8")
        record.signal = str(signal)
        snapshot = cost_tracker.snapshot()
        record.metrics = {
            "llm_calls": snapshot.llm_calls,
            "tool_calls": snapshot.tool_calls,
            "input_tokens": snapshot.input_tokens,
            "output_tokens": snapshot.output_tokens,
        }
        record.status = "completed"
        record.emit(
            "run.completed",
            {"signal": record.signal, "metrics": record.metrics},
        )


def _runtime_config(
    request: CreateRunRequest,
) -> tuple[dict[str, Any], str, str, tuple[str, ...]]:
    ticker = normalize_symbol(request.ticker)
    asset_type = "crypto" if ticker.endswith(CRYPTO_SUFFIXES) else "stock"
    if asset_type == "crypto" and "fundamentals" in request.analysts:
        raise ValueError("fundamentals analyst is unavailable for crypto assets")
    if request.orchestration_mode == "learned" and not DEFAULT_CONFIG.get(
        "scheduler_adapter_path"
    ):
        raise ValueError("local learned scheduler requires a configured adapter path")
    config = DEFAULT_CONFIG.copy()
    rounds = DEPTH_ROUNDS[request.research_depth]
    config.update(
        {
            "orchestration_mode": request.orchestration_mode,
            "output_language": request.output_language,
            "max_debate_rounds": rounds,
            "max_risk_discuss_rounds": rounds,
            "scheduler_fallback_enabled": request.orchestration_mode == "static",
        }
    )
    analysts = tuple(key for key in ANALYST_ORDER if key in request.analysts)
    return config, ticker, asset_type, analysts


def _node_kind(node_name: str) -> str:
    if node_name == "Scheduler":
        return "scheduler"
    if node_name.startswith("tools_"):
        return "tool"
    return "agent"


def _report_sections(state: dict[str, Any]) -> dict[str, str]:
    reports = {
        key: str(state.get(key) or "")
        for key in REPORT_FIELDS
        if key not in {"investment_plan", "final_trade_decision"}
    }
    research = state.get("investment_debate_state") or {}
    reports["investment_plan"] = str(
        state.get("investment_plan") or research.get("judge_decision") or ""
    )
    risk = state.get("risk_debate_state") or {}
    reports["final_trade_decision"] = str(
        state.get("final_trade_decision") or risk.get("judge_decision") or ""
    )
    return reports
