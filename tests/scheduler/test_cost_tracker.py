from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tradingagents.scheduler.cost_tracker import SchedulerCostCallback


def test_cost_tracker_reports_deltas_for_tokens_and_tools() -> None:
    tracker = SchedulerCostCallback()
    before = tracker.snapshot()
    tracker.on_chat_model_start({}, [[]])
    tracker.on_tool_start({}, "input")
    message = AIMessage(
        content="done",
        usage_metadata={"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
    )
    tracker.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))

    after = tracker.snapshot()
    cost = after.delta(before)
    assert after.llm_calls == 1
    assert cost.tool_calls == 1
    assert cost.input_tokens == 100
    assert cost.output_tokens == 25


def test_cost_tracker_ignores_unrecognized_response_shape() -> None:
    tracker = SchedulerCostCallback()
    tracker.on_llm_end(SimpleNamespace(generations=[]))
    assert tracker.snapshot().input_tokens == 0
