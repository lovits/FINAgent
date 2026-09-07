from __future__ import annotations

import re
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
from tradingagents.scheduler.cost_tracker import CostSnapshot, SchedulerCostCallback

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

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
TRACE_CONTENT_LIMIT = 20_000
TRACE_MESSAGE_LIMIT = 24
METRIC_KEYS = ("llm_calls", "tool_calls", "input_tokens", "output_tokens")


class RunCancelled(RuntimeError):
    pass

ANALYST_REPORT_BY_NODE = {
    "Market Analyst": "market_report",
    "Sentiment Analyst": "sentiment_report",
    "News Analyst": "news_report",
    "Fundamentals Analyst": "fundamentals_report",
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
    error_details: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, int] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    resolved_ticker: str | None = None
    data_sources: dict[str, str] = field(default_factory=dict)
    active_node: str | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event, repr=False)
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
            if len(self.events) <= cursor + 1 and self.status not in TERMINAL_STATUSES:
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
            "error_details": dict(self.error_details),
            "metrics": dict(self.metrics),
            "models": dict(self.models),
            "resolved_ticker": self.resolved_ticker,
            "data_sources": dict(self.data_sources),
            "event_count": len(self.events),
        }


class GraphEventProjector:
    def __init__(
        self,
        record: RunRecord,
        cost_tracker: SchedulerCostCallback | None = None,
    ):
        self.record = record
        self.cost_tracker = cost_tracker
        self._reports: dict[str, str] = {}
        self._last_expert_snapshot = CostSnapshot()
        self._scheduler_metrics = _empty_metrics()

    def __call__(self, stream_mode: str, payload: object) -> None:
        if self.record.cancel_requested.is_set():
            raise RunCancelled("analysis cancelled by user")
        if stream_mode == "updates" and isinstance(payload, dict):
            self._project_nodes(payload)
        elif stream_mode == "values" and isinstance(payload, dict):
            self._project_reports(payload)

    def _project_nodes(self, payload: dict[str, Any]) -> None:
        visible = [
            (node_name, update)
            for node_name, update in payload.items()
            if not node_name.startswith("Msg Clear")
        ]
        if not visible:
            return
        expert_snapshot = (
            self.cost_tracker.snapshot() if self.cost_tracker else CostSnapshot()
        )
        expert_delta = _snapshot_delta(expert_snapshot, self._last_expert_snapshot)
        self._last_expert_snapshot = expert_snapshot
        for index, (node_name, update) in enumerate(visible):
            self.record.active_node = node_name
            usage = expert_delta if index == 0 else _empty_metrics()
            if node_name == "Scheduler":
                scheduler_usage = _scheduler_usage(update)
                self._scheduler_metrics = _add_metrics(
                    self._scheduler_metrics, scheduler_usage
                )
                usage = _add_metrics(usage, scheduler_usage)
            cumulative = self.metrics_snapshot(expert_snapshot)
            data = {
                "node": node_name,
                "kind": _node_kind(node_name),
                "status": _node_status(node_name, update),
                "timestamp_ms": _now_ms(),
                "usage": usage,
                "cumulative_metrics": cumulative,
            }
            data.update(_node_details(update))
            if isinstance(update, dict) and update.get("scheduler_action"):
                data["selected_action"] = update["scheduler_action"]
                data["valid_actions"] = update.get("scheduler_valid_actions", [])
                data["scheduler_step"] = update.get("scheduler_step")
                data["scheduler_history"] = update.get("scheduler_history", [])
                data["policy_id"] = update.get("scheduler_policy_id")
            self.record.emit("node.progress", data)
            self.record.metrics = dict(cumulative)

    def metrics_snapshot(
        self, expert_snapshot: CostSnapshot | None = None
    ) -> dict[str, int]:
        current = expert_snapshot or (
            self.cost_tracker.snapshot() if self.cost_tracker else CostSnapshot()
        )
        expert = {
            "llm_calls": current.llm_calls,
            "tool_calls": current.tool_calls,
            "input_tokens": current.input_tokens,
            "output_tokens": current.output_tokens,
        }
        return {
            key: int(expert[key]) + int(self._scheduler_metrics[key])
            for key in METRIC_KEYS
        }

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
        settings_provider: Callable[[], dict[str, object]] | None = None,
    ):
        self.graph_factory = graph_factory
        self.settings_provider = settings_provider or (lambda: {})
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="web-run")
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}

    def create(self, request: CreateRunRequest) -> RunRecord:
        with self._lock:
            if any(run.status in {"queued", "running"} for run in self._runs.values()):
                raise RuntimeError("another analysis is already running")
            resolved_ticker = normalize_symbol(request.ticker)
            record = RunRecord(
                uuid.uuid4().hex,
                request,
                resolved_ticker=resolved_ticker,
                data_sources=_data_source_summary(resolved_ticker),
            )
            self._runs = {record.run_id: record}
            self._executor.submit(self._execute, record)
            return record

    def get(self, run_id: str) -> RunRecord:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown run: {run_id}") from exc

    def cancel(self, run_id: str) -> RunRecord:
        record = self.get(run_id)
        if record.status in TERMINAL_STATUSES:
            return record
        record.status = "cancelling"
        record.cancel_requested.set()
        record.emit("run.cancelling", {"run_id": run_id})
        return record

    def _execute(self, record: RunRecord) -> None:
        record.status = "running"
        record.emit("run.started", {"run_id": record.run_id})
        try:
            self._run_graph(record)
        except RunCancelled:
            record.status = "cancelled"
            record.emit("run.cancelled", {"run_id": record.run_id})
        except Exception as exc:  # noqa: BLE001 - boundary converts failure to run state
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            record.error_details = _error_details(exc, record.active_node)
            record.emit(
                "run.failed",
                {"error": record.error, "error_details": record.error_details},
            )

    def _run_graph(self, record: RunRecord) -> None:
        config, ticker, asset_type, analysts = _runtime_config(
            record.request,
            self.settings_provider(),
        )
        record.models = _model_summary(config, record.request.orchestration_mode)
        if record.cancel_requested.is_set():
            raise RunCancelled("analysis cancelled before execution")
        cost_tracker = SchedulerCostCallback()
        projector = GraphEventProjector(record, cost_tracker)
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
            on_graph_event=projector,
        )
        destination = Path(config["results_dir"]) / "web" / record.run_id
        report_path = graph.save_reports(final_state, ticker, destination)
        record.complete_report = report_path.read_text(encoding="utf-8")
        record.signal = str(signal)
        record.metrics = projector.metrics_snapshot()
        record.status = "completed"
        record.emit(
            "run.completed",
            {"signal": record.signal, "metrics": record.metrics},
        )


def _runtime_config(
    request: CreateRunRequest,
    overrides: dict[str, object] | None = None,
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
    config.update(overrides or {})
    rounds = DEPTH_ROUNDS[request.research_depth]
    config.update(
        {
            "orchestration_mode": request.orchestration_mode,
            "output_language": request.output_language,
            "max_debate_rounds": rounds,
            "max_risk_discuss_rounds": rounds,
            "llm_max_retries": 1,
            "checkpoint_enabled": False,
            "scheduler_fallback_enabled": request.orchestration_mode == "static",
        }
    )
    analysts = tuple(key for key in ANALYST_ORDER if key in request.analysts)
    return config, ticker, asset_type, analysts


def _model_summary(config: dict[str, Any], mode: str) -> dict[str, str]:
    base_model = str(config.get("scheduler_base_model") or "Local scheduler")
    adapter = str(config.get("scheduler_adapter_path") or "")
    scheduler = {
        "static": "Static LangGraph",
        "teacher": str(config.get("teacher_model") or "Teacher model"),
        "learned": f"{Path(base_model).name} + {_adapter_name(adapter)}",
    }[mode]
    return {
        "expert": str(config.get("quick_think_llm") or "Configured expert model"),
        "scheduler": scheduler,
    }


def _adapter_name(value: str) -> str:
    path = Path(value)
    if path.name == "checkpoint" and path.parent.name.startswith("round-"):
        return f"RL{path.parent.name.removeprefix('round-')}"
    return path.name or "LoRA"


def _node_kind(node_name: str) -> str:
    if node_name == "Scheduler":
        return "scheduler"
    if node_name.startswith("tools_"):
        return "tool"
    return "agent"


def _node_status(node_name: str, update: object) -> str:
    report_field = ANALYST_REPORT_BY_NODE.get(node_name)
    if report_field and isinstance(update, dict) and not update.get(report_field):
        return "running"
    return "completed"


def _node_details(update: object) -> dict[str, object]:
    if not isinstance(update, dict):
        return {}
    produced_fields = [key for key in REPORT_FIELDS if update.get(key)]
    messages = update.get("messages") or []
    message_records = _message_records(messages)
    message = message_records[-1]["content"] if message_records else None
    tool_calls = _tool_calls(messages)
    return {
        "produced_fields": produced_fields,
        "message": message,
        "messages": message_records,
        "tool_calls": tool_calls,
    }


def _message_records(messages: object) -> list[dict[str, object]]:
    if not isinstance(messages, (list, tuple)):
        return []
    result = []
    for value in messages[-TRACE_MESSAGE_LIMIT:]:
        content = (
            value.get("content")
            if isinstance(value, dict)
            else getattr(value, "content", None)
        )
        if content is None:
            continue
        text = content if isinstance(content, str) else str(content)
        message_type = (
            value.get("type")
            if isinstance(value, dict)
            else getattr(value, "type", type(value).__name__)
        )
        is_tool_message = str(message_type).lower() == "tool"
        display_text = _tool_result_summary(text) if is_tool_message else text[:TRACE_CONTENT_LIMIT]
        record: dict[str, object] = {
            "type": str(message_type or type(value).__name__),
            "content": display_text,
            "content_length": len(text),
            "truncated": not is_tool_message and len(text) > TRACE_CONTENT_LIMIT,
        }
        if is_tool_message:
            record["summarized"] = True
            source = _tool_result_source(text)
            if source:
                record["source"] = source
        for key in ("name", "tool_call_id"):
            item = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
            if item:
                record[key] = str(item)
        result.append(record)
    return result


def _tool_calls(messages: object) -> list[dict[str, object]]:
    if not isinstance(messages, (list, tuple)) or not messages:
        return []
    value = messages[-1]
    calls = value.get("tool_calls", []) if isinstance(value, dict) else getattr(value, "tool_calls", [])
    result = []
    for call in calls or []:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
        identifier = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
        safe_args = _safe_value(args)
        argument_keys = sorted(safe_args) if isinstance(safe_args, dict) else []
        item = {"name": str(name or "unknown"), "argument_keys": argument_keys}
        if identifier:
            item["id"] = str(identifier)
        result.append(item)
    return result


def _tool_result_summary(text: str) -> str:
    """Compact a potentially huge tool payload into source/result metadata."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "工具已完成，但没有返回正文。"
    selected = [lines[0]]
    for pattern in (r"^#?\s*Source:", r"^#?\s*Total records:", r"^DATA_UNAVAILABLE", r"^NO_DATA_AVAILABLE"):
        match = next((line for line in lines if re.search(pattern, line, re.IGNORECASE)), None)
        if match and match not in selected:
            selected.append(match)
    selected.append(f"原始返回共 {len(text):,} 个字符，界面已省略明细。")
    return "\n".join(selected)


def _tool_result_source(text: str) -> str | None:
    match = re.search(r"^#?\s*Source:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def _data_source_summary(ticker: str) -> dict[str, str]:
    from tradingagents.dataflows.symbol_utils import is_a_share_symbol

    if is_a_share_symbol(ticker):
        return {
            "market": "BaoStock",
            "fundamentals": "BaoStock",
            "news": "AKShare / Eastmoney",
        }
    return {"market": "Configured global vendor", "news": "Configured global vendor"}


def _error_details(exc: Exception, active_node: str | None) -> dict[str, str]:
    error_type = type(exc).__name__
    message = str(exc).strip() or "未返回具体错误正文"
    lower = f"{error_type} {message}".lower()
    if "rate limit" in lower or "too many requests" in lower:
        suggestion = "数据源触发限流。A股请使用 MOUTAI、PINGAN、CATL、BYD 等英文别名；其他市场稍后重试。"
    elif "no data" in lower or "market data" in lower:
        suggestion = "请检查股票别名和分析日期；非交易日会自动使用最近交易日，但无覆盖标的仍会失败。"
    elif "teacher" in lower or "openrouter" in lower:
        suggestion = "请检查 OpenRouter Key、模型名和账户限额，然后重新创建任务。"
    else:
        suggestion = "请保留该错误信息并重新创建任务；若再次失败，可根据节点和错误类型继续定位。"
    return {
        "node": active_node or "任务初始化",
        "type": error_type,
        "message": message,
        "suggestion": suggestion,
    }


def _safe_value(value: object) -> object:
    """Convert tool metadata to JSON-safe primitives before exposing its keys."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    return str(value)


def _empty_metrics() -> dict[str, int]:
    return dict.fromkeys(METRIC_KEYS, 0)


def _snapshot_delta(current: CostSnapshot, previous: CostSnapshot) -> dict[str, int]:
    return {
        "llm_calls": max(0, current.llm_calls - previous.llm_calls),
        "tool_calls": max(0, current.tool_calls - previous.tool_calls),
        "input_tokens": max(0, current.input_tokens - previous.input_tokens),
        "output_tokens": max(0, current.output_tokens - previous.output_tokens),
    }


def _scheduler_usage(update: object) -> dict[str, int]:
    if not isinstance(update, dict):
        return _empty_metrics()
    metadata = update.get("scheduler_decision_metadata")
    usage = metadata.get("usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return _empty_metrics()
    return {
        key: _nonnegative_int(usage.get(key))
        for key in METRIC_KEYS
    }


def _nonnegative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _add_metrics(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(left[key]) + int(right[key]) for key in METRIC_KEYS}


def _report_sections(state: dict[str, Any]) -> dict[str, str]:
    reports = {
        key: str(state.get(key) or "")
        for key in REPORT_FIELDS
        if key not in {"investment_plan", "final_trade_decision"}
    }
    research = state.get("investment_debate_state") or {}
    reports["investment_plan"] = str(
        state.get("investment_plan")
        or _joined_report(
            ("Bull Researcher", research.get("bull_history")),
            ("Bear Researcher", research.get("bear_history")),
            ("Research Manager", research.get("judge_decision")),
        )
    )
    risk = state.get("risk_debate_state") or {}
    reports["final_trade_decision"] = str(
        state.get("final_trade_decision")
        or _joined_report(
            ("Aggressive Analyst", risk.get("aggressive_history")),
            ("Conservative Analyst", risk.get("conservative_history")),
            ("Neutral Analyst", risk.get("neutral_history")),
            ("Portfolio Manager", risk.get("judge_decision")),
        )
    )
    return reports


def _joined_report(*parts: tuple[str, object]) -> str:
    return "\n\n".join(
        f"### {title}\n{content}"
        for title, content in parts
        if isinstance(content, str) and content.strip()
    )
