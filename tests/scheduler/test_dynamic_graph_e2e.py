from langchain_core.messages import AIMessage

import tradingagents.graph.setup as graph_setup_module
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup
from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import CallableSchedulerPolicy


def _factory(update):
    return lambda llm: lambda state: update(state) if callable(update) else update


def _policy(context):
    state = context.state
    if not state.get("market_report"):
        return SchedulerAction.MARKET
    if not state["investment_debate_state"].get("history"):
        return SchedulerAction.BULL
    if not state.get("investment_plan"):
        return SchedulerAction.RESEARCH_MANAGER
    if not state.get("trader_investment_plan"):
        return SchedulerAction.TRADER
    if not state["risk_debate_state"].get("history"):
        return SchedulerAction.AGGRESSIVE
    if not state.get("final_trade_decision"):
        return SchedulerAction.PORTFOLIO_MANAGER
    return SchedulerAction.STOP


def _initial_state() -> dict:
    return {
        "messages": [AIMessage(content="start")],
        "company_of_interest": "AAPL",
        "asset_type": "stock",
        "instrument_context": "Apple Inc.",
        "trade_date": "2026-01-05",
        "sender": "human",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "latest_speaker": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "final_trade_decision": "",
        "past_context": "",
    }


def test_real_langgraph_executes_dynamic_agent_path(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_setup_module,
        "create_market_analyst",
        _factory(
            {
                "market_report": "market evidence",
                "messages": [AIMessage(content="market complete")],
            }
        ),
    )
    monkeypatch.setattr(graph_setup_module, "create_msg_delete", lambda: lambda state: {})
    monkeypatch.setattr(
        graph_setup_module,
        "create_bull_researcher",
        _factory(
            {
                "investment_debate_state": {
                    "history": "Bull: evidence",
                    "bull_history": "Bull: evidence",
                    "bear_history": "",
                    "current_response": "Bull: evidence",
                    "judge_decision": "",
                    "count": 1,
                }
            }
        ),
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_research_manager",
        _factory({"investment_plan": "Hold with risk controls"}),
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_trader",
        _factory({"trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD"}),
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_aggressive_debator",
        _factory(
            {
                "risk_debate_state": {
                    "history": "Aggressive: upside",
                    "aggressive_history": "Aggressive: upside",
                    "conservative_history": "",
                    "neutral_history": "",
                    "latest_speaker": "Aggressive",
                    "current_aggressive_response": "Aggressive: upside",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 1,
                }
            }
        ),
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_portfolio_manager",
        _factory({"final_trade_decision": "**Rating**: Hold"}),
    )

    graph_setup = GraphSetup(
        object(),
        object(),
        {key: (lambda state: {}) for key in ("market", "social", "news", "fundamentals")},
        ConditionalLogic(),
    )
    workflow = graph_setup.setup_scheduler_graph(
        ("market",),
        CallableSchedulerPolicy(_policy, policy_id="dynamic-test"),
    )
    final_state = workflow.compile().invoke(
        _initial_state(),
        config={"recursion_limit": 30},
    )

    assert final_state["final_trade_decision"] == "**Rating**: Hold"
    assert final_state["scheduler_history"] == [
        "<ACT_MARKET>",
        "<ACT_BULL>",
        "<ACT_RESEARCH_MANAGER>",
        "<ACT_TRADER>",
        "<ACT_AGGRESSIVE>",
        "<ACT_PORTFOLIO_MANAGER>",
        "<ACT_STOP>",
    ]
    assert "<ACT_BEAR>" not in final_state["scheduler_history"]
    assert "<ACT_CONSERVATIVE>" not in final_state["scheduler_history"]
