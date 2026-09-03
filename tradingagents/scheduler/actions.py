"""Stable action vocabulary for the central agent scheduler."""

from __future__ import annotations

from enum import Enum

ACTION_SCHEMA_VERSION = "scheduler-actions-v1"


class SchedulerAction(str, Enum):
    MARKET = "<ACT_MARKET>"
    SENTIMENT = "<ACT_SENTIMENT>"
    NEWS = "<ACT_NEWS>"
    FUNDAMENTALS = "<ACT_FUNDAMENTALS>"
    BULL = "<ACT_BULL>"
    BEAR = "<ACT_BEAR>"
    RESEARCH_MANAGER = "<ACT_RESEARCH_MANAGER>"
    TRADER = "<ACT_TRADER>"
    AGGRESSIVE = "<ACT_AGGRESSIVE>"
    CONSERVATIVE = "<ACT_CONSERVATIVE>"
    NEUTRAL = "<ACT_NEUTRAL>"
    PORTFOLIO_MANAGER = "<ACT_PORTFOLIO_MANAGER>"
    STOP = "<ACT_STOP>"


NODE_BY_ACTION: dict[SchedulerAction, str] = {
    SchedulerAction.MARKET: "Market Analyst",
    SchedulerAction.SENTIMENT: "Sentiment Analyst",
    SchedulerAction.NEWS: "News Analyst",
    SchedulerAction.FUNDAMENTALS: "Fundamentals Analyst",
    SchedulerAction.BULL: "Bull Researcher",
    SchedulerAction.BEAR: "Bear Researcher",
    SchedulerAction.RESEARCH_MANAGER: "Research Manager",
    SchedulerAction.TRADER: "Trader",
    SchedulerAction.AGGRESSIVE: "Aggressive Analyst",
    SchedulerAction.CONSERVATIVE: "Conservative Analyst",
    SchedulerAction.NEUTRAL: "Neutral Analyst",
    SchedulerAction.PORTFOLIO_MANAGER: "Portfolio Manager",
}

ACTION_BY_NODE = {node: action for action, node in NODE_BY_ACTION.items()}

ANALYST_ACTION_BY_KEY: dict[str, SchedulerAction] = {
    "market": SchedulerAction.MARKET,
    "social": SchedulerAction.SENTIMENT,
    "news": SchedulerAction.NEWS,
    "fundamentals": SchedulerAction.FUNDAMENTALS,
}


def parse_action(value: SchedulerAction | str) -> SchedulerAction:
    """Return a canonical scheduler action or raise a useful validation error."""

    if isinstance(value, SchedulerAction):
        return value
    if not isinstance(value, str):
        raise ValueError(f"scheduler action must be a string, got {type(value).__name__}")
    try:
        return SchedulerAction(value.strip())
    except ValueError as exc:
        raise ValueError(f"unknown scheduler action: {value!r}") from exc


def node_for_action(action: SchedulerAction | str) -> str:
    """Resolve a non-terminal action to its original LangGraph node name."""

    parsed = parse_action(action)
    if parsed is SchedulerAction.STOP:
        raise ValueError("STOP does not map to an Expert Agent node")
    return NODE_BY_ACTION[parsed]
