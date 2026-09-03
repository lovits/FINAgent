"""Thread-safe LangChain callback counters for scheduler trajectory costs."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

from .trajectory import ExecutionCost


@dataclass(frozen=True)
class CostSnapshot:
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def delta(self, previous: CostSnapshot) -> ExecutionCost:
        return ExecutionCost(
            tool_calls=max(0, self.tool_calls - previous.tool_calls),
            input_tokens=max(0, self.input_tokens - previous.input_tokens),
            output_tokens=max(0, self.output_tokens - previous.output_tokens),
        )


class SchedulerCostCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._snapshot = CostSnapshot()
        self._tool_events: list[dict[str, Any]] = []

    def on_llm_start(self, serialized, prompts, **kwargs: Any) -> None:
        self._increment(llm_calls=1)

    def on_chat_model_start(self, serialized, messages, **kwargs: Any) -> None:
        self._increment(llm_calls=1)

    def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs: Any) -> None:
        with self._lock:
            current = self._snapshot
            self._snapshot = CostSnapshot(
                llm_calls=current.llm_calls,
                tool_calls=current.tool_calls + 1,
                input_tokens=current.input_tokens,
                output_tokens=current.output_tokens,
            )
            self._tool_events.append(
                {
                    "tool_call_id": str(run_id) if run_id is not None else None,
                    "tool_name": self._tool_name(serialized, kwargs),
                    "status": "running",
                }
            )

    def on_tool_end(self, output, *, run_id=None, **kwargs: Any) -> None:
        self._finish_tool(run_id, status="succeeded")

    def on_tool_error(self, error, *, run_id=None, **kwargs: Any) -> None:
        self._finish_tool(
            run_id,
            status="failed",
            error_type=type(error).__name__,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
            message = generation.message
        except (AttributeError, IndexError, TypeError):
            return
        usage = message.usage_metadata if isinstance(message, AIMessage) else None
        if usage:
            self._increment(
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
            )

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return self._snapshot

    def tool_events_since(self, index: int) -> list[dict[str, Any]]:
        if index < 0:
            raise ValueError("tool event index cannot be negative")
        with self._lock:
            return [dict(event) for event in self._tool_events[index:]]

    def _finish_tool(
        self,
        run_id: object,
        *,
        status: str,
        error_type: str | None = None,
    ) -> None:
        identifier = str(run_id) if run_id is not None else None
        with self._lock:
            for event in reversed(self._tool_events):
                if event["status"] != "running":
                    continue
                if identifier is not None and event["tool_call_id"] != identifier:
                    continue
                event["status"] = status
                if error_type is not None:
                    event["error_type"] = error_type
                return

    @staticmethod
    def _tool_name(serialized: object, kwargs: Mapping[str, Any]) -> str:
        if isinstance(serialized, Mapping):
            value = serialized.get("name") or serialized.get("id")
            if value:
                return str(value)
        value = kwargs.get("name")
        return str(value) if value else "unknown"

    def _increment(
        self,
        *,
        llm_calls: int = 0,
        tool_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self._lock:
            current = self._snapshot
            self._snapshot = CostSnapshot(
                llm_calls=current.llm_calls + llm_calls,
                tool_calls=current.tool_calls + tool_calls,
                input_tokens=current.input_tokens + input_tokens,
                output_tokens=current.output_tokens + output_tokens,
            )
