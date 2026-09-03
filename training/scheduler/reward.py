"""Outcome-and-cost reward for complete scheduler trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .audit import parse_portfolio_rating, parse_trader_action
from .pair import PORTFOLIO_RATINGS


@dataclass(frozen=True)
class RewardConfig:
    portfolio_quality_weight: float = 0.70
    trader_quality_weight: float = 0.30
    format_compliance_weight: float = 0.10
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
    format_compliance: float
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


def portfolio_similarity(candidate: object, reference: object) -> float:
    candidate_rating = parse_portfolio_rating(candidate)
    reference_rating = parse_portfolio_rating(reference)
    if candidate_rating not in PORTFOLIO_RATINGS or reference_rating not in PORTFOLIO_RATINGS:
        return 0.0
    distance = abs(
        PORTFOLIO_RATINGS.index(candidate_rating)
        - PORTFOLIO_RATINGS.index(reference_rating)
    )
    return 1 - distance / (len(PORTFOLIO_RATINGS) - 1)


def score_trajectory(
    trajectory: SchedulerTrajectory,
    static_reference: SchedulerTrajectory,
    *,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    config = config or RewardConfig()
    if trajectory.task_id != static_reference.task_id:
        raise ValueError("reward requires a Static reference from the same task")
    if trajectory.data_snapshot_id != static_reference.data_snapshot_id:
        raise ValueError("reward requires the same data snapshot")
    fallback_active = trajectory.execution_status == "fallback"
    outputs = trajectory.final_outputs
    reference_outputs = static_reference.final_outputs
    trader_action = parse_trader_action(outputs.get("trader_investment_plan"))
    reference_action = parse_trader_action(
        reference_outputs.get("trader_investment_plan")
    )
    complete = (
        not fallback_active
        and trajectory.execution_status == "completed"
        and bool(str(outputs.get("investment_plan") or "").strip())
        and trader_action is not None
        and parse_portfolio_rating(outputs.get("final_trade_decision")) is not None
    )
    portfolio_quality = (
        portfolio_similarity(
            outputs.get("final_trade_decision"),
            reference_outputs.get("final_trade_decision"),
        )
        * config.portfolio_quality_weight
        if complete
        else 0.0
    )
    trader_quality = (
        config.trader_quality_weight
        if complete and trader_action == reference_action
        else 0.0
    )
    format_ok = (
        bool(trajectory.steps)
        and trajectory.steps[-1].selected_action == SchedulerAction.STOP.value
        and sum(
            step.selected_action == SchedulerAction.STOP.value
            for step in trajectory.steps
        )
        == 1
        and not any(step.error for step in trajectory.steps)
    )
    format_compliance = config.format_compliance_weight if format_ok else 0.0
    completion = config.completion_bonus if complete else 0.0

    cost = trajectory.cost_total
    agent_cost = cost.agent_calls * config.agent_call_cost
    tool_cost = cost.tool_calls * config.tool_call_cost
    token_cost = (
        (cost.input_tokens + cost.output_tokens) / 1000 * config.token_cost_per_1k
    )
    latency_cost = cost.latency_ms / 1000 * config.latency_cost_per_second
    no_progress_count = sum(
        step.selected_action != SchedulerAction.STOP.value
        and step.state_before == step.state_after
        for step in trajectory.steps
    )
    no_progress = no_progress_count * config.no_progress_penalty
    invalid = (
        config.invalid_penalty
        if any(
            step.error or step.selected_action not in step.valid_actions
            for step in trajectory.steps
        )
        else 0.0
    )
    incomplete = 0.0 if complete else config.incomplete_penalty
    fallback = config.fallback_penalty if fallback_active else 0.0
    total = (
        portfolio_quality
        + trader_quality
        + format_compliance
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
        total,
        portfolio_quality,
        trader_quality,
        format_compliance,
        completion,
        agent_cost,
        tool_cost,
        token_cost,
        latency_cost,
        no_progress,
        invalid,
        incomplete,
        fallback,
    )
