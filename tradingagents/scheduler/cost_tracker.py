"""Small mutable accumulator for trajectory-level execution cost."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


@dataclass
class CostTracker:
    agent_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(
        self,
        *,
        agent_calls: int = 0,
        tool_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        retries: int = 0,
    ) -> None:
        for name, value in {
            "agent_calls": agent_calls,
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "retries": retries,
        }.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        self.agent_calls += agent_calls
        self.tool_calls += tool_calls
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.latency_ms += latency_ms
        self.retries += retries

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["total_tokens"] = self.total_tokens
        return result


class SchedulerLLMStatsCallback(BaseCallbackHandler):
    """Track cumulative LLM tokens without recording prompts or credentials."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return
        message = getattr(generation, "message", None)
        usage = message.usage_metadata if isinstance(message, AIMessage) else None
        if usage:
            with self._lock:
                self.input_tokens += int(usage.get("input_tokens", 0))
                self.output_tokens += int(usage.get("output_tokens", 0))

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }


class SchedulerToolStatsCallback(BaseCallbackHandler):
    """Track cumulative ToolNode calls separately from LLM callbacks."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.tool_calls = 0

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        with self._lock:
            self.tool_calls += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"tool_calls": self.tool_calls}
