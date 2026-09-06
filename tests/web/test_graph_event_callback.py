from functools import partial
from unittest.mock import MagicMock

from tradingagents.graph.trading_graph import TradingAgentsGraph


def test_run_graph_forwards_stream_events_without_changing_finalization() -> None:
    final_state = {
        "company_of_interest": "NVDA",
        "trade_date": "2026-08-20",
        "final_trade_decision": "Rating: Hold",
    }
    graph = MagicMock()
    graph.orchestration_mode = "static"
    graph.scheduler_fallback_reason = None
    graph.debug = False
    graph.callbacks = []
    graph.config = {"checkpoint_enabled": False}
    graph.selected_analysts = ("market",)
    graph.propagator.create_initial_state.return_value = {"messages": []}
    graph.propagator.get_graph_args.return_value = {}
    graph.graph.stream.return_value = [
        ("updates", {"Market Analyst": {"market_report": "Market"}}),
        ("values", final_state),
    ]
    graph.process_signal.return_value = "HOLD"
    graph._run_graph = partial(TradingAgentsGraph._run_graph, graph)
    events = []

    result = graph._run_graph(
        "NVDA",
        "2026-08-20",
        on_graph_event=lambda mode, payload: events.append((mode, payload)),
    )

    assert result == (final_state, "HOLD")
    assert [mode for mode, _ in events] == ["updates", "values"]
    graph._log_state.assert_called_once_with("2026-08-20", final_state)
