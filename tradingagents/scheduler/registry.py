"""Versioned descriptions of Expert Agents exposed to scheduler policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .actions import ANALYST_ACTION_BY_KEY, SchedulerAction

AGENT_CATALOG_VERSION = "agent-catalog-v1"


@dataclass(frozen=True)
class AgentSpec:
    key: str
    action: SchedulerAction
    node_name: str
    purpose: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    prerequisites: tuple[str, ...]
    completion_signal: str
    tool_policy_owner: Literal["expert_agent"] = "expert_agent"

    def to_prompt_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["action"] = self.action.value
        return value


_AGENT_SPECS = (
    AgentSpec(
        "market",
        SchedulerAction.MARKET,
        "Market Analyst",
        "Analyze price action, technical indicators, and verified market data.",
        ("company_of_interest", "trade_date", "instrument_context"),
        ("market_report",),
        (),
        "market_report is non-empty",
    ),
    AgentSpec(
        "social",
        SchedulerAction.SENTIMENT,
        "Sentiment Analyst",
        "Analyze market sentiment from news and social sources.",
        ("company_of_interest", "trade_date", "instrument_context"),
        ("sentiment_report",),
        (),
        "sentiment_report is non-empty",
    ),
    AgentSpec(
        "news",
        SchedulerAction.NEWS,
        "News Analyst",
        "Analyze company, macroeconomic, insider, and prediction-market events.",
        ("company_of_interest", "trade_date", "instrument_context"),
        ("news_report",),
        (),
        "news_report is non-empty",
    ),
    AgentSpec(
        "fundamentals",
        SchedulerAction.FUNDAMENTALS,
        "Fundamentals Analyst",
        "Analyze financial statements and company fundamentals.",
        ("company_of_interest", "trade_date", "instrument_context"),
        ("fundamentals_report",),
        (),
        "fundamentals_report is non-empty",
    ),
    AgentSpec(
        "bull",
        SchedulerAction.BULL,
        "Bull Researcher",
        "Build and defend the bullish investment case from available reports.",
        ("investment_debate_state",),
        ("investment_debate_state",),
        ("at least one analyst report",),
        "investment_debate_state contains a Bull response",
    ),
    AgentSpec(
        "bear",
        SchedulerAction.BEAR,
        "Bear Researcher",
        "Build and defend the bearish investment case from available reports.",
        ("investment_debate_state",),
        ("investment_debate_state",),
        ("at least one analyst report",),
        "investment_debate_state contains a Bear response",
    ),
    AgentSpec(
        "research_manager",
        SchedulerAction.RESEARCH_MANAGER,
        "Research Manager",
        "Judge the research debate and produce an actionable investment plan.",
        ("investment_debate_state",),
        ("investment_plan", "investment_debate_state"),
        ("investment_debate_state.history is non-empty",),
        "investment_plan is non-empty",
    ),
    AgentSpec(
        "trader",
        SchedulerAction.TRADER,
        "Trader",
        "Turn research into a concrete transaction proposal.",
        ("investment_plan",),
        ("trader_investment_plan",),
        ("investment_plan is non-empty",),
        "trader_investment_plan is non-empty",
    ),
    AgentSpec(
        "aggressive",
        SchedulerAction.AGGRESSIVE,
        "Aggressive Analyst",
        "Assess the transaction from a high-risk, high-upside perspective.",
        ("trader_investment_plan", "risk_debate_state"),
        ("risk_debate_state",),
        ("trader_investment_plan is non-empty",),
        "risk_debate_state contains an Aggressive response",
    ),
    AgentSpec(
        "conservative",
        SchedulerAction.CONSERVATIVE,
        "Conservative Analyst",
        "Assess the transaction from a capital-preservation perspective.",
        ("trader_investment_plan", "risk_debate_state"),
        ("risk_debate_state",),
        ("trader_investment_plan is non-empty",),
        "risk_debate_state contains a Conservative response",
    ),
    AgentSpec(
        "neutral",
        SchedulerAction.NEUTRAL,
        "Neutral Analyst",
        "Balance upside and downside in the transaction risk assessment.",
        ("trader_investment_plan", "risk_debate_state"),
        ("risk_debate_state",),
        ("trader_investment_plan is non-empty",),
        "risk_debate_state contains a Neutral response",
    ),
    AgentSpec(
        "portfolio_manager",
        SchedulerAction.PORTFOLIO_MANAGER,
        "Portfolio Manager",
        "Synthesize the risk debate into the final portfolio decision.",
        ("investment_plan", "trader_investment_plan", "risk_debate_state"),
        ("final_trade_decision", "risk_debate_state"),
        ("risk_debate_state.history is non-empty",),
        "final_trade_decision is non-empty",
    ),
)

AGENT_REGISTRY = {spec.action: spec for spec in _AGENT_SPECS}


def registry_for_analysts(selected_analysts: tuple[str, ...]) -> tuple[AgentSpec, ...]:
    """Return the active analyst specs plus every downstream Expert Agent."""

    if not selected_analysts:
        raise ValueError("at least one analyst must be selected")
    unknown = set(selected_analysts) - set(ANALYST_ACTION_BY_KEY)
    if unknown:
        raise ValueError(f"unknown analyst keys: {sorted(unknown)}")
    if len(set(selected_analysts)) != len(selected_analysts):
        raise ValueError("selected analyst keys must be unique")

    active_analyst_actions = {
        ANALYST_ACTION_BY_KEY[analyst_key] for analyst_key in selected_analysts
    }
    return tuple(
        spec
        for spec in _AGENT_SPECS
        if spec.action not in ANALYST_ACTION_BY_KEY.values()
        or spec.action in active_analyst_actions
    )
