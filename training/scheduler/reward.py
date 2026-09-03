"""Deterministic trajectory reward for agent orchestration."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from tradingagents.agents.utils.rating import RATINGS_5_TIER, parse_rating
from tradingagents.scheduler.trajectory import SchedulerTrajectory

_TRADER_ACTION_RE = re.compile(
    r"FINAL\s+TRANSACTION\s+PROPOSAL\s*:\s*\*{0,2}(BUY|HOLD|SELL)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RewardConfig:
    portfolio_quality_weight: float = 0.70
    trader_quality_weight: float = 0.30
    completion_bonus: float = 0.20
    agent_call_cost: float = 0.02
    tool_call_cost: float = 0.005
    token_cost_per_1k: float = 0.005
    latency_cost_per_second: float = 0.0
    no_progress_penalty: float = 0.10
    invalid_penalty: float = 1.0
    incomplete_penalty: float = 1.0
    fallback_penalty: float = 0.25
    min_reward: float = -1.5
    max_reward: float = 1.2


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    portfolio_quality: float
    trader_quality: float
    completion: float
    agent_cost: float
    tool_cost: float
    token_cost: float
    latency_cost: float
    no_progress: float
    invalid: float
    incomplete: float
    fallback: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_trader_action(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _TRADER_ACTION_RE.search(value)
    if match:
        return match.group(1).title()
    lowered = value.lower()
    for action in ("buy", "hold", "sell"):
        if re.search(rf"\b{action}\b", lowered):
            return action.title()
    return None


def portfolio_similarity(candidate: str, reference: str) -> float:
    candidate_rating = parse_rating(candidate)
    reference_rating = parse_rating(reference)
    candidate_index = RATINGS_5_TIER.index(candidate_rating)
    reference_index = RATINGS_5_TIER.index(reference_rating)
    return 1.0 - abs(candidate_index - reference_index) / (len(RATINGS_5_TIER) - 1)


def score_trajectory(
    trajectory: SchedulerTrajectory,
    final_state: dict[str, Any],
    static_state: dict[str, Any],
    *,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Score completion/quality first, then subtract measurable execution cost."""

    config = config or RewardConfig()
    candidate_pm = str(final_state.get("final_trade_decision") or "")
    reference_pm = str(static_state.get("final_trade_decision") or "")
    candidate_trader = str(final_state.get("trader_investment_plan") or "")
    reference_trader = str(static_state.get("trader_investment_plan") or "")
    complete = all(
        _text(final_state.get(field))
        for field in ("investment_plan", "trader_investment_plan", "final_trade_decision")
    )

    portfolio_quality = (
        portfolio_similarity(candidate_pm, reference_pm)
        * config.portfolio_quality_weight
        if complete and _text(reference_pm)
        else 0.0
    )
    candidate_action = parse_trader_action(candidate_trader)
    reference_action = parse_trader_action(reference_trader)
    trader_quality = (
        config.trader_quality_weight
        if complete and candidate_action is not None and candidate_action == reference_action
        else 0.0
    )
    completion = config.completion_bonus if complete else 0.0

    agent_calls = sum(1 for step in trajectory.steps if step.agent_node)
    tool_calls = sum(step.tool_calls for step in trajectory.steps)
    tokens = sum(step.input_tokens + step.output_tokens for step in trajectory.steps)
    latency_ms = sum(step.latency_ms for step in trajectory.steps)
    agent_cost = agent_calls * config.agent_call_cost
    tool_cost = tool_calls * config.tool_call_cost
    token_cost = tokens / 1000 * config.token_cost_per_1k
    latency_cost = latency_ms / 1000 * config.latency_cost_per_second
    no_progress = sum(step.no_progress for step in trajectory.steps) * config.no_progress_penalty
    invalid = config.invalid_penalty if any(step.error for step in trajectory.steps) else 0.0
    incomplete = 0.0 if complete else config.incomplete_penalty
    fallback = config.fallback_penalty if trajectory.fallback_reason else 0.0

    total = (
        portfolio_quality
        + trader_quality
        + completion
        - agent_cost
        - tool_cost
        - token_cost
        - latency_cost
        - no_progress
        - invalid
        - incomplete
        - fallback
    )
    total = max(config.min_reward, min(config.max_reward, total))
    return RewardBreakdown(
        total=total,
        portfolio_quality=portfolio_quality,
        trader_quality=trader_quality,
        completion=completion,
        agent_cost=agent_cost,
        tool_cost=tool_cost,
        token_cost=token_cost,
        latency_cost=latency_cost,
        no_progress=no_progress,
        invalid=invalid,
        incomplete=incomplete,
        fallback=fallback,
    )
