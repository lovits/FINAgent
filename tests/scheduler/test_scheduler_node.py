import pytest
from langchain_core.messages import HumanMessage

from tradingagents.graph import setup as graph_setup_module
from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import CallableSchedulerPolicy, StaticSchedulerPolicy
from tradingagents.scheduler.scheduler_node import SchedulerFallbackError, SchedulerNode


def _initial_state():
    return {
        "messages": [HumanMessage(content="Analyze")],
        "company_of_interest": "NVDA",
        "asset_type": "stock",
        "instrument_context": "NVIDIA Corp",
        "trade_date": "2025-01-02",
        "sender": "user",
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "history": "",
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


def test_scheduler_node_retries_one_invalid_policy_decision():
    calls = 0

    def selector(context):
        nonlocal calls
        calls += 1
        return SchedulerAction.TRADER if calls == 1 else SchedulerAction.MARKET

    node = SchedulerNode(CallableSchedulerPolicy(selector), ("market",))
    update = node(_initial_state())

    assert calls == 2
    assert update["scheduler_action"] == SchedulerAction.MARKET.value


def test_scheduler_node_raises_typed_fallback_after_retries():
    node = SchedulerNode(
        CallableSchedulerPolicy(lambda context: SchedulerAction.TRADER),
        ("market",),
    )

    with pytest.raises(SchedulerFallbackError, match="invalid_policy_decision"):
        node(_initial_state())


class _LearnedConditionalLogic:
    max_debate_rounds = 1
    max_risk_discuss_rounds = 1

    def should_continue_market(self, state):
        return "Msg Clear Market"


def _debate_update(speaker):
    def node(state):
        current = state["investment_debate_state"]
        count = current["count"] + 1
        response = f"{speaker}: case"
        return {
            "investment_debate_state": {
                **current,
                "history": current["history"] + response,
                "current_response": response,
                "count": count,
            }
        }

    return node


def _risk_update(speaker):
    def node(state):
        current = state["risk_debate_state"]
        return {
            "risk_debate_state": {
                **current,
                "history": current["history"] + f"{speaker}: risk",
                "latest_speaker": speaker,
                "count": current["count"] + 1,
            }
        }

    return node


def test_learned_graph_runs_end_to_end_with_mock_experts(monkeypatch):
    monkeypatch.setattr(
        graph_setup_module,
        "create_market_analyst",
        lambda llm: lambda state: {"market_report": "market complete"},
    )
    monkeypatch.setattr(graph_setup_module, "create_msg_delete", lambda: lambda state: {})
    monkeypatch.setattr(
        graph_setup_module, "create_bull_researcher", lambda llm: _debate_update("Bull")
    )
    monkeypatch.setattr(
        graph_setup_module, "create_bear_researcher", lambda llm: _debate_update("Bear")
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_research_manager",
        lambda llm: lambda state: {"investment_plan": "research plan"},
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_trader",
        lambda llm: lambda state: {"trader_investment_plan": "Buy"},
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_aggressive_debator",
        lambda llm: _risk_update("Aggressive"),
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_conservative_debator",
        lambda llm: _risk_update("Conservative"),
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_neutral_debator",
        lambda llm: _risk_update("Neutral"),
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_portfolio_manager",
        lambda llm: lambda state: {"final_trade_decision": "Overweight"},
    )

    setup = graph_setup_module.GraphSetup(
        None,
        None,
        {"market": lambda state: {}},
        _LearnedConditionalLogic(),
    )
    workflow = setup.setup_learned_graph(
        ("market",), StaticSchedulerPolicy(), scheduler_max_steps=16
    )
    result = workflow.compile().invoke(_initial_state(), {"recursion_limit": 50})

    assert result["final_trade_decision"] == "Overweight"
    assert result["scheduler_history"] == [
        "<ACT_MARKET>",
        "<ACT_BULL>",
        "<ACT_BEAR>",
        "<ACT_RESEARCH_MANAGER>",
        "<ACT_TRADER>",
        "<ACT_AGGRESSIVE>",
        "<ACT_CONSERVATIVE>",
        "<ACT_NEUTRAL>",
        "<ACT_PORTFOLIO_MANAGER>",
        "<ACT_STOP>",
    ]
    assert result["scheduler_agent_calls"] == 9
