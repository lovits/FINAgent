"""Pair independently accepted Static and Teacher trajectories by task snapshot."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from tradingagents.scheduler.trajectory import ExecutionCost, SchedulerTrajectory

from .audit import parse_portfolio_rating, parse_trader_action

PAIRING_VERSION = "scheduler-pair-v1"
PORTFOLIO_RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")


@dataclass(frozen=True)
class PairComparison:
    task_id: str
    data_snapshot_id: str | None
    static_trajectory_id: str
    teacher_trajectory_id: str
    trader_action_match: bool
    portfolio_rating_distance: int | None
    teacher_to_static_cost_ratio: float | None
    pair_status: str
    rejection_reasons: tuple[str, ...]
    pairing_version: str = PAIRING_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def pair_trajectories(
    static: SchedulerTrajectory,
    teacher: SchedulerTrajectory,
    *,
    max_rating_distance: int = 1,
    max_cost_ratio: float = 1.5,
) -> PairComparison:
    if static.task_id != teacher.task_id:
        raise ValueError("cannot pair trajectories from different tasks")
    if static.data_snapshot_id != teacher.data_snapshot_id:
        raise ValueError("cannot pair trajectories from different data snapshots")

    reasons = []
    if static.audit_status not in {"accepted", "warning"}:
        reasons.append("static_not_accepted")
    if teacher.audit_status not in {"accepted", "warning"}:
        reasons.append("teacher_not_accepted")

    for field in (
        "task_dataset_version",
        "information_cutoff",
        "selected_analysts",
        "research_depth",
        "output_language",
        "expert_config_hash",
        "scheduler_max_steps",
    ):
        static_value = static.provenance.get(field)
        teacher_value = teacher.provenance.get(field)
        if static_value is None or teacher_value is None or static_value != teacher_value:
            reasons.append(f"provenance_mismatch:{field}")

    static_trader = parse_trader_action(
        static.final_outputs.get("trader_investment_plan")
    )
    teacher_trader = parse_trader_action(
        teacher.final_outputs.get("trader_investment_plan")
    )
    trader_match = static_trader is not None and static_trader == teacher_trader
    if not trader_match:
        reasons.append("trader_action_mismatch")

    rating_distance = _rating_distance(static, teacher)
    if rating_distance is None or rating_distance > max_rating_distance:
        reasons.append("portfolio_rating_too_far")

    cost_ratio = _cost_ratio(static.cost_total, teacher.cost_total)
    if cost_ratio is None:
        reasons.append("teacher_cost_uncomparable")
    elif cost_ratio > max_cost_ratio:
        reasons.append("teacher_cost_too_high")

    return PairComparison(
        task_id=static.task_id,
        data_snapshot_id=static.data_snapshot_id,
        static_trajectory_id=static.trajectory_id,
        teacher_trajectory_id=teacher.trajectory_id,
        trader_action_match=trader_match,
        portfolio_rating_distance=rating_distance,
        teacher_to_static_cost_ratio=cost_ratio,
        pair_status="teacher_verified" if not reasons else "rejected",
        rejection_reasons=tuple(reasons),
    )


def _rating_distance(
    static: SchedulerTrajectory, teacher: SchedulerTrajectory
) -> int | None:
    static_rating = parse_portfolio_rating(static.final_outputs.get("final_trade_decision"))
    teacher_rating = parse_portfolio_rating(
        teacher.final_outputs.get("final_trade_decision")
    )
    if static_rating not in PORTFOLIO_RATINGS or teacher_rating not in PORTFOLIO_RATINGS:
        return None
    return abs(PORTFOLIO_RATINGS.index(static_rating) - PORTFOLIO_RATINGS.index(teacher_rating))


def _cost_value(cost: ExecutionCost) -> float:
    tokens = cost.input_tokens + cost.output_tokens
    return 0.02 * cost.agent_calls + 0.005 * cost.tool_calls + 0.005 * tokens / 1000


def _cost_ratio(static: ExecutionCost, teacher: ExecutionCost) -> float | None:
    static_value = _cost_value(static)
    teacher_value = _cost_value(teacher)
    if static_value == 0:
        return 1.0 if teacher_value == 0 else None
    return teacher_value / static_value
