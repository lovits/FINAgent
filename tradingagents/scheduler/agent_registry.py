"""Single source of truth for schedulable TradingAgents experts."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import SchedulerAction

AGENT_REGISTRY_VERSION = "v1"


@dataclass(frozen=True)
class AgentSpec:
    key: str
    action: SchedulerAction
    node_name: str
    purpose: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    prerequisites: tuple[str, ...] = ()
    internal_tools: tuple[str, ...] = ()
    analyst_key: str | None = None

    @property
    def tool_policy_owner(self) -> str:
        return "expert_agent"


AGENT_REGISTRY: dict[SchedulerAction, AgentSpec] = {
    SchedulerAction.MARKET: AgentSpec(
        key="market",
        action=SchedulerAction.MARKET,
        node_name="Market Analyst",
        purpose="Analyze price, volume, indicators, and the verified market snapshot.",
        reads=("company_of_interest", "trade_date", "instrument_context"),
        writes=("market_report",),
        internal_tools=("get_stock_data", "get_indicators", "get_verified_market_snapshot"),
        analyst_key="market",
    ),
    SchedulerAction.SENTIMENT: AgentSpec(
        key="sentiment",
        action=SchedulerAction.SENTIMENT,
        node_name="Sentiment Analyst",
        purpose="Analyze news and social sentiment evidence.",
        reads=("company_of_interest", "trade_date", "instrument_context"),
        writes=("sentiment_report",),
        internal_tools=("get_news",),
        analyst_key="social",
    ),
    SchedulerAction.NEWS: AgentSpec(
        key="news",
        action=SchedulerAction.NEWS,
        node_name="News Analyst",
        purpose="Analyze company, global, macro, insider, and prediction-market news.",
        reads=("company_of_interest", "trade_date", "instrument_context"),
        writes=("news_report",),
        internal_tools=(
            "get_news",
            "get_global_news",
            "get_insider_transactions",
            "get_macro_indicators",
            "get_prediction_markets",
        ),
        analyst_key="news",
    ),
    SchedulerAction.FUNDAMENTALS: AgentSpec(
        key="fundamentals",
        action=SchedulerAction.FUNDAMENTALS,
        node_name="Fundamentals Analyst",
        purpose="Analyze fundamentals, balance sheet, cashflow, and income statement.",
        reads=("company_of_interest", "trade_date", "instrument_context"),
        writes=("fundamentals_report",),
        internal_tools=(
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
        ),
        analyst_key="fundamentals",
    ),
    SchedulerAction.BULL: AgentSpec(
        key="bull",
        action=SchedulerAction.BULL,
        node_name="Bull Researcher",
        purpose="Build and defend the bullish investment case from available reports.",
        reads=(
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "investment_debate_state",
        ),
        writes=("investment_debate_state",),
    ),
    SchedulerAction.BEAR: AgentSpec(
        key="bear",
        action=SchedulerAction.BEAR,
        node_name="Bear Researcher",
        purpose="Build and defend the bearish investment case from available reports.",
        reads=(
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "investment_debate_state",
        ),
        writes=("investment_debate_state",),
    ),
    SchedulerAction.RESEARCH_MANAGER: AgentSpec(
        key="research_manager",
        action=SchedulerAction.RESEARCH_MANAGER,
        node_name="Research Manager",
        purpose="Judge the research debate and produce the investment plan.",
        reads=("investment_debate_state",),
        writes=("investment_plan",),
        prerequisites=("investment_debate_state.history",),
    ),
    SchedulerAction.TRADER: AgentSpec(
        key="trader",
        action=SchedulerAction.TRADER,
        node_name="Trader",
        purpose="Convert the research plan into a Buy, Hold, or Sell proposal.",
        reads=("investment_plan",),
        writes=("trader_investment_plan",),
        prerequisites=("investment_plan",),
    ),
    SchedulerAction.AGGRESSIVE: AgentSpec(
        key="aggressive",
        action=SchedulerAction.AGGRESSIVE,
        node_name="Aggressive Analyst",
        purpose="Assess the proposal from an aggressive risk perspective.",
        reads=("trader_investment_plan", "risk_debate_state"),
        writes=("risk_debate_state",),
        prerequisites=("trader_investment_plan",),
    ),
    SchedulerAction.CONSERVATIVE: AgentSpec(
        key="conservative",
        action=SchedulerAction.CONSERVATIVE,
        node_name="Conservative Analyst",
        purpose="Assess downside and capital-preservation risks.",
        reads=("trader_investment_plan", "risk_debate_state"),
        writes=("risk_debate_state",),
        prerequisites=("trader_investment_plan",),
    ),
    SchedulerAction.NEUTRAL: AgentSpec(
        key="neutral",
        action=SchedulerAction.NEUTRAL,
        node_name="Neutral Analyst",
        purpose="Balance aggressive and conservative risk arguments.",
        reads=("trader_investment_plan", "risk_debate_state"),
        writes=("risk_debate_state",),
        prerequisites=("trader_investment_plan",),
    ),
    SchedulerAction.PORTFOLIO_MANAGER: AgentSpec(
        key="portfolio_manager",
        action=SchedulerAction.PORTFOLIO_MANAGER,
        node_name="Portfolio Manager",
        purpose="Produce the final portfolio decision from research, trade, and risk evidence.",
        reads=("trader_investment_plan", "risk_debate_state", "past_context"),
        writes=("final_trade_decision",),
        prerequisites=("trader_investment_plan", "risk_debate_state.history"),
    ),
}


def registry_for_analysts(selected_analysts: tuple[str, ...] | list[str]) -> dict[SchedulerAction, AgentSpec]:
    """Return experts available for the selected analyst configuration."""

    selected = set(selected_analysts)
    return {
        action: spec
        for action, spec in AGENT_REGISTRY.items()
        if spec.analyst_key is None or spec.analyst_key in selected
    }
