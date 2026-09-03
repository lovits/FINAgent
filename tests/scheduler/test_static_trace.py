from typing import TypedDict

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph

from tradingagents.scheduler.actions import SchedulerAction
from training.scheduler.static_trace import capture_static_graph


class _TraceState(TypedDict, total=False):
    company_of_interest: str
    trade_date: str
    asset_type: str
    market_pass: int
    tool_seen: bool
    messages: list
    sender: str
    market_report: str
    investment_debate_state: dict
    investment_plan: str
    trader_investment_plan: str
    risk_debate_state: dict
    final_trade_decision: str


def _build_static_graph():
    workflow = StateGraph(_TraceState)

    def market(state):
        count = int(state.get("market_pass", 0)) + 1
        update = {"market_pass": count, "sender": "Market Analyst"}
        if count == 2:
            update["market_report"] = "Verified price and indicator evidence."
        return update

    workflow.add_node("Market Analyst", market)
    workflow.add_node(
        "tools_market",
        lambda _state: {
            "tool_seen": True,
            "messages": [
                ToolMessage(
                    content="Verified OHLCV data",
                    name="get_stock_data",
                    tool_call_id="call-1",
                )
            ],
        },
    )
    workflow.add_node("Msg Clear Market", lambda _state: {"sender": "Msg Clear Market"})
    workflow.add_node(
        "Bull Researcher",
        lambda _state: {
            "investment_debate_state": {
                "count": 1,
                "history": "Bull: upside evidence",
                "current_response": "Bull: upside evidence",
            }
        },
    )
    workflow.add_node(
        "Bear Researcher",
        lambda _state: {
            "investment_debate_state": {
                "count": 2,
                "history": "Bull: upside evidence\nBear: downside evidence",
                "current_response": "Bear: downside evidence",
            }
        },
    )
    workflow.add_node(
        "Research Manager", lambda _state: {"investment_plan": "Balanced plan"}
    )
    workflow.add_node(
        "Trader",
        lambda _state: {
            "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**"
        },
    )
    workflow.add_node(
        "Aggressive Analyst",
        lambda _state: {
            "risk_debate_state": {
                "count": 1,
                "history": "Aggressive: accept risk",
                "latest_speaker": "Aggressive Analyst",
            }
        },
    )
    workflow.add_node(
        "Conservative Analyst",
        lambda _state: {
            "risk_debate_state": {
                "count": 2,
                "history": "Aggressive: accept risk\nConservative: reduce exposure",
                "latest_speaker": "Conservative Analyst",
            }
        },
    )
    workflow.add_node(
        "Neutral Analyst",
        lambda _state: {
            "risk_debate_state": {
                "count": 3,
                "history": "Aggressive, Conservative, Neutral risk views",
                "latest_speaker": "Neutral Analyst",
            }
        },
    )
    workflow.add_node(
        "Portfolio Manager",
        lambda _state: {
            "final_trade_decision": "The Buy case was rejected. **Rating**: Sell"
        },
    )

    workflow.add_edge(START, "Market Analyst")
    workflow.add_conditional_edges(
        "Market Analyst",
        lambda state: "tool" if state["market_pass"] == 1 else "clear",
        {"tool": "tools_market", "clear": "Msg Clear Market"},
    )
    workflow.add_edge("tools_market", "Market Analyst")
    sequence = [
        "Msg Clear Market",
        "Bull Researcher",
        "Bear Researcher",
        "Research Manager",
        "Trader",
        "Aggressive Analyst",
        "Conservative Analyst",
        "Neutral Analyst",
        "Portfolio Manager",
    ]
    for source, target in zip(sequence, sequence[1:], strict=False):
        workflow.add_edge(source, target)
    workflow.add_edge("Portfolio Manager", END)
    return workflow.compile()


def test_capture_original_static_graph_includes_nodes_and_scheduler_examples():
    record = capture_static_graph(
        _build_static_graph(),
        {
            "company_of_interest": "NVDA",
            "trade_date": "2026-08-26",
            "asset_type": "stock",
        },
        graph_config={},
        task={
            "task_id": "NVDA:2026-08-26:stock",
            "ticker": "NVDA",
            "trade_date": "2026-08-26",
            "seed_family": "earnings_window",
        },
        selected_analysts=("market",),
        max_debate_rounds=1,
        max_risk_rounds=1,
        provenance={"graph_mode": "static", "information_cutoff": "2026-08-26"},
    )

    assert record["status"] == "accepted"
    assert [step["node"] for step in record["node_steps"]][:4] == [
        "Market Analyst",
        "tools_market",
        "Market Analyst",
        "Msg Clear Market",
    ]
    assert record["node_steps"][1]["node_type"] == "tool_node"
    assert record["node_steps"][1]["tool_events"] == [
        {
            "name": "get_stock_data",
            "tool_call_id": "call-1",
            "status": "success",
            "success": True,
            "data_available": True,
            "output_summary": "Verified OHLCV data",
        }
    ]
    assert record["node_steps"][3]["node_type"] == "control_node"

    actions = [step["target_action"] for step in record["scheduler_examples"]]
    assert actions == [
        SchedulerAction.MARKET.value,
        SchedulerAction.BULL.value,
        SchedulerAction.BEAR.value,
        SchedulerAction.RESEARCH_MANAGER.value,
        SchedulerAction.TRADER.value,
        SchedulerAction.AGGRESSIVE.value,
        SchedulerAction.CONSERVATIVE.value,
        SchedulerAction.NEUTRAL.value,
        SchedulerAction.PORTFOLIO_MANAGER.value,
        SchedulerAction.STOP.value,
    ]
    assert record["agent_sequence"].count("Market Analyst") == 1
    assert record["final_outputs"] == {
        "trader_action": "Buy",
        "portfolio_rating": "Sell",
    }
    assert record["labels"]["data_sparse"] is False
    assert record["labels"]["high_risk"] is True
    assert record["quality_control"]["static_actions_valid"] is True


def test_failed_static_run_is_preserved_as_rejected_record():
    workflow = StateGraph(_TraceState)

    def fail(_state):
        raise RuntimeError("fixture tool outage")

    workflow.add_node("Market Analyst", fail)
    workflow.add_edge(START, "Market Analyst")
    workflow.add_edge("Market Analyst", END)
    record = capture_static_graph(
        workflow.compile(),
        {
            "company_of_interest": "NVDA",
            "trade_date": "2026-08-26",
            "asset_type": "stock",
        },
        graph_config={},
        task={"task_id": "failed", "ticker": "NVDA", "trade_date": "2026-08-26"},
        selected_analysts=("market",),
        max_debate_rounds=1,
        max_risk_rounds=1,
        provenance={"graph_mode": "static"},
    )

    assert record["status"] == "rejected"
    assert "fixture tool outage" in record["quality_control"]["failure_reason"]
