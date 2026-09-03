"""Capture the original Static LangGraph as auditable dataset records."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Any

from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.scheduler.action_mask import compute_action_mask
from tradingagents.scheduler.actions import (
    ACTION_BY_NODE,
    ACTION_SCHEMA_VERSION,
    SchedulerAction,
)
from tradingagents.scheduler.cost_tracker import (
    SchedulerLLMStatsCallback,
    SchedulerToolStatsCallback,
)
from tradingagents.scheduler.state_serializer import (
    STATE_SCHEMA_VERSION,
    serialize_scheduler_state,
)

from .reward import parse_portfolio_rating, parse_trader_action

STATIC_DATASET_VERSION = "v1"
STATIC_POLICY_ID = "original-static-langgraph-v1"
_SPARSE_MARKERS = (
    "<unavailable>",
    "not available",
    "insufficient data",
    "n/a",
    "data_unavailable",
    "no_data_available",
)


def _bounded(value: Any, limit: int = 2000) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _message_summary(message: Any) -> dict[str, Any]:
    tool_calls = getattr(message, "tool_calls", None) or []
    return {
        "type": type(message).__name__,
        "content": _bounded(getattr(message, "content", ""), 500),
        "tool_calls": [
            call.get("name", "") if isinstance(call, dict) else getattr(call, "name", "")
            for call in tool_calls
        ],
    }


def snapshot_state(state: Mapping[str, Any]) -> dict[str, Any]:
    debate = state.get("investment_debate_state") or {}
    risk = state.get("risk_debate_state") or {}
    messages = state.get("messages") or []
    return {
        "company_of_interest": state.get("company_of_interest", ""),
        "trade_date": state.get("trade_date", ""),
        "sender": state.get("sender", ""),
        "reports": {
            name: _bounded(state.get(name))
            for name in (
                "market_report",
                "sentiment_report",
                "news_report",
                "fundamentals_report",
            )
        },
        "research": {
            "count": int(debate.get("count") or 0),
            "current_response": _bounded(debate.get("current_response")),
            "history": _bounded(debate.get("history")),
            "investment_plan": _bounded(state.get("investment_plan")),
        },
        "trade": _bounded(state.get("trader_investment_plan")),
        "risk": {
            "count": int(risk.get("count") or 0),
            "latest_speaker": risk.get("latest_speaker", ""),
            "history": _bounded(risk.get("history")),
            "final_trade_decision": _bounded(state.get("final_trade_decision")),
        },
        "recent_messages": [_message_summary(message) for message in messages[-3:]],
    }


def _node_type(node: str) -> str:
    if node in ACTION_BY_NODE:
        return "expert_agent"
    if node.startswith("tools_"):
        return "tool_node"
    if node.startswith("Msg Clear"):
        return "control_node"
    return "graph_node"


def _is_internal_analyst_return(node: str, previous_node: str | None) -> bool:
    for spec in ANALYST_NODE_SPECS.values():
        if node == spec.agent_node:
            return previous_node == spec.tool_node
    return False


def _stats_delta(current: Mapping[str, int], previous: Mapping[str, int]) -> dict[str, int]:
    return {
        key: max(0, int(current.get(key, 0)) - int(previous.get(key, 0)))
        for key in ("llm_calls", "tool_calls", "input_tokens", "output_tokens")
    }


def _message_field(message: Any, field: str, default: Any = "") -> Any:
    if isinstance(message, Mapping):
        return message.get(field, default)
    return getattr(message, field, default)


def _tool_events(update: Any) -> list[dict[str, Any]]:
    if not isinstance(update, Mapping):
        return []
    messages = update.get("messages") or []
    if not isinstance(messages, (list, tuple)):
        messages = [messages]
    events = []
    for message in messages:
        message_type = str(_message_field(message, "type", type(message).__name__)).lower()
        if message_type not in {"tool", "toolmessage"}:
            continue
        content = str(_message_field(message, "content", "") or "")
        status = str(_message_field(message, "status", "") or "completed").lower()
        lowered = content.lower()
        explicit_error = status == "error" or lowered.lstrip().startswith("error")
        events.append(
            {
                "name": str(_message_field(message, "name", "") or ""),
                "tool_call_id": str(
                    _message_field(message, "tool_call_id", "") or ""
                ),
                "status": "error" if explicit_error else status,
                "success": not explicit_error,
                "data_available": not any(marker in lowered for marker in _SPARSE_MARKERS),
                "output_summary": _bounded(content, 500),
            }
        )
    return events


def capture_static_graph(
    compiled_graph: Any,
    initial_state: dict[str, Any],
    *,
    graph_config: dict[str, Any],
    task: Mapping[str, Any],
    selected_analysts: Sequence[str],
    max_debate_rounds: int,
    max_risk_rounds: int,
    provenance: Mapping[str, Any],
    stats_provider: Callable[[], dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Run the actual static graph and capture node updates plus full state snapshots."""

    previous_state: dict[str, Any] | None = None
    previous_node: str | None = None
    previous_stats = stats_provider() if stats_provider else {}
    step_started = monotonic()
    pending: list[dict[str, Any]] = []
    node_steps: list[dict[str, Any]] = []
    scheduler_examples: list[dict[str, Any]] = []
    scheduler_step = 0
    static_actions_valid = True

    failure_reason = None
    try:
        stream = compiled_graph.stream(
            initial_state,
            config=graph_config,
            stream_mode=["updates", "values"],
        )
        for mode, payload in stream:
            if mode == "updates":
                current_stats = stats_provider() if stats_provider else {}
                for node, update in payload.items():
                    pending.append(
                        {
                            "node": node,
                            "node_type": _node_type(node),
                            "update_keys": (
                                sorted(update) if isinstance(update, Mapping) else []
                            ),
                            "state_before": snapshot_state(previous_state or initial_state),
                            "latency_ms": max(
                                0.0, (monotonic() - step_started) * 1000
                            ),
                            "cost_delta": _stats_delta(current_stats, previous_stats),
                            "tool_events": _tool_events(update),
                        }
                    )
                previous_stats = current_stats
                continue

            current_state = dict(payload)
            if previous_state is None:
                previous_state = deepcopy(current_state)
                step_started = monotonic()
                continue

            after = snapshot_state(current_state)
            for item in pending:
                item["step_id"] = len(node_steps)
                item["state_after"] = after
                node_steps.append(item)
                node = item["node"]
                if node in ACTION_BY_NODE and not _is_internal_analyst_return(
                    node, previous_node
                ):
                    action = ACTION_BY_NODE[node]
                    mask = compute_action_mask(
                        previous_state,
                        selected_analysts,
                        step=scheduler_step,
                        max_steps=32,
                        max_debate_rounds=max_debate_rounds,
                        max_risk_rounds=max_risk_rounds,
                    )
                    action_valid = action in mask.valid_actions
                    static_actions_valid = static_actions_valid and action_valid
                    scheduler_examples.append(
                        {
                            "step_id": scheduler_step,
                            "input_text": serialize_scheduler_state(
                                previous_state,
                                mask.valid_actions,
                                step=scheduler_step,
                                max_steps=32,
                            ),
                            "valid_actions": [value.value for value in mask.valid_actions],
                            "target_action": action.value,
                            "source_node": node,
                            "action_valid": action_valid,
                        }
                    )
                    scheduler_step += 1
                previous_node = node
            pending.clear()
            previous_state = deepcopy(current_state)
            step_started = monotonic()
    except Exception as exc:  # one failed task must still yield an auditable record
        failure_reason = f"{type(exc).__name__}: {_bounded(exc, 500)}"
        for item in pending:
            item["step_id"] = len(node_steps)
            item["state_after"] = snapshot_state(previous_state or initial_state)
            item["error"] = failure_reason
            node_steps.append(item)
        pending.clear()

    final_state = previous_state or initial_state
    stop_mask = compute_action_mask(
        final_state,
        selected_analysts,
        step=scheduler_step,
        max_steps=32,
        max_debate_rounds=max_debate_rounds,
        max_risk_rounds=max_risk_rounds,
    )
    if SchedulerAction.STOP in stop_mask.valid_actions:
        scheduler_examples.append(
            {
                "step_id": scheduler_step,
                "input_text": serialize_scheduler_state(
                    final_state,
                    stop_mask.valid_actions,
                    step=scheduler_step,
                    max_steps=32,
                ),
                "valid_actions": [value.value for value in stop_mask.valid_actions],
                "target_action": SchedulerAction.STOP.value,
                "source_node": "END",
                "action_valid": True,
            }
        )
    else:
        static_actions_valid = False

    reports = {
        field: str(final_state.get(field) or "")
        for field in (
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
        )
    }
    trader_text = str(final_state.get("trader_investment_plan") or "")
    decision_text = str(final_state.get("final_trade_decision") or "")
    trader_action = parse_trader_action(trader_text)
    portfolio_rating = parse_portfolio_rating(decision_text)
    completed = bool(final_state.get("investment_plan")) and bool(trader_action) and bool(
        portfolio_rating
    )
    selected_report_fields = [
        ANALYST_NODE_SPECS[key].report_key
        for key in selected_analysts
        if key in ANALYST_NODE_SPECS
    ]
    selected_reports = [str(final_state.get(field) or "") for field in selected_report_fields]
    data_sparse = any(not text.strip() for text in selected_reports) or any(
        marker in " ".join(selected_reports).lower() for marker in _SPARSE_MARKERS
    )
    final_stats = stats_provider() if stats_provider else {}
    tool_events = [
        event for step in node_steps for event in step.get("tool_events", [])
    ]
    tool_failure = any(
        not event["success"] or not event["data_available"] for event in tool_events
    )
    data_sparse = data_sparse or tool_failure
    accepted = completed and static_actions_valid and failure_reason is None
    return {
        "record_type": "static_langgraph",
        "dataset_version": STATIC_DATASET_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "policy_id": STATIC_POLICY_ID,
        "status": "accepted" if accepted else "rejected",
        "task": dict(task),
        "provenance": dict(provenance),
        "agent_sequence": [
            item["source_node"]
            for item in scheduler_examples
            if item["target_action"] != SchedulerAction.STOP.value
        ],
        "node_steps": node_steps,
        "scheduler_examples": scheduler_examples,
        "reports": reports,
        "investment_plan": str(final_state.get("investment_plan") or ""),
        "trader_result": trader_text,
        "final_decision": decision_text,
        "final_outputs": {
            "trader_action": trader_action,
            "portfolio_rating": portfolio_rating,
        },
        "cost": {
            **final_stats,
            "agent_calls": sum(
                item["target_action"] != SchedulerAction.STOP.value
                for item in scheduler_examples
            ),
            "latency_ms": sum(step["latency_ms"] for step in node_steps),
        },
        "labels": {
            "completed": completed,
            "evidence_conflict": None,
            "data_sparse": data_sparse,
            "high_risk": task.get("seed_family") == "high_volatility"
            or portfolio_rating in {"Underweight", "Sell"},
        },
        "quality_control": {
            "static_actions_valid": static_actions_valid,
            "human_review": "pending",
            "tool_events": len(tool_events),
            "tool_failures_or_missing_data": sum(
                not event["success"] or not event["data_available"]
                for event in tool_events
            ),
            "failure_reason": (
                None
                if accepted
                else failure_reason or "incomplete_or_invalid_static_trace"
            ),
        },
    }


class StaticLangGraphRunner:
    """Create one dataset record by running the original static graph."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        selected_analysts: Sequence[str],
        graph_factory: Callable[..., TradingAgentsGraph] = TradingAgentsGraph,
    ):
        self.config = dict(config, scheduler_mode="static")
        self.selected_analysts = tuple(selected_analysts)
        self.graph_factory = graph_factory

    @staticmethod
    def _git_commit() -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        llm_stats = SchedulerLLMStatsCallback()
        tool_stats = SchedulerToolStatsCallback()
        graph = self.graph_factory(
            self.selected_analysts,
            config=self.config,
            debug=False,
            callbacks=[llm_stats],
        )
        ticker = str(task["ticker"])
        trade_date = str(task["trade_date"])
        asset_type = str(task.get("asset_type", "stock"))
        initial_state = graph.propagator.create_initial_state(
            ticker,
            trade_date,
            asset_type=asset_type,
            past_context=str(task.get("past_context", "")),
            instrument_context=graph.resolve_instrument_context(ticker, asset_type),
        )
        graph_config = graph.propagator.get_graph_args(callbacks=[tool_stats])["config"]

        def stats() -> dict[str, int]:
            return {**llm_stats.snapshot(), **tool_stats.snapshot()}

        return capture_static_graph(
            graph.graph,
            initial_state,
            graph_config=graph_config,
            task=task,
            selected_analysts=self.selected_analysts,
            max_debate_rounds=int(self.config["max_debate_rounds"]),
            max_risk_rounds=int(self.config["max_risk_discuss_rounds"]),
            provenance={
                "code_commit": self._git_commit(),
                "graph_mode": "static",
                "policy_id": STATIC_POLICY_ID,
                "expert_prompts": "repository_prompts_at_code_commit",
                "llm_provider": self.config["llm_provider"],
                "quick_model": self.config["quick_think_llm"],
                "deep_model": self.config["deep_think_llm"],
                "data_vendors": self.config.get("data_vendors", {}),
                "information_cutoff": trade_date,
            },
            stats_provider=stats,
        )


def append_static_record(record: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
