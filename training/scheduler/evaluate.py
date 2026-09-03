"""Fair task-aligned metrics for Static, Teacher, SFT, and GRPO trajectories."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .audit import parse_portfolio_rating, parse_trader_action
from .pair import PORTFOLIO_RATINGS


def evaluate_trajectories(
    candidates: Iterable[SchedulerTrajectory],
    static_references: Mapping[tuple[str, str | None], SchedulerTrajectory],
) -> dict[str, Any]:
    values = list(candidates)
    if not values:
        raise ValueError("evaluation requires at least one candidate trajectory")
    task_metrics = [
        _trajectory_metrics(
            trajectory,
            static_references.get((trajectory.task_id, trajectory.data_snapshot_id)),
        )
        for trajectory in values
    ]
    paths = {
        tuple(step.selected_action for step in trajectory.steps) for trajectory in values
    }
    return {
        "trajectory_count": len(values),
        "completion_rate": mean(value["completed"] for value in task_metrics),
        "failure_rate": mean(value["failed"] for value in task_metrics),
        "legal_action_rate": mean(value["legal"] for value in task_metrics),
        "invalid_action_rate": mean(not value["legal"] for value in task_metrics),
        "stop_correct_rate": mean(value["stop_correct"] for value in task_metrics),
        "trader_parse_rate": mean(value["trader_parse"] for value in task_metrics),
        "portfolio_parse_rate": mean(value["portfolio_parse"] for value in task_metrics),
        "fallback_rate": mean(value["fallback"] for value in task_metrics),
        "budget_exhausted_rate": mean(
            value["budget_exhausted"] for value in task_metrics
        ),
        "context_overflow_rate": mean(
            value["context_overflow"] for value in task_metrics
        ),
        "trader_match_rate": mean(value["trader_match"] for value in task_metrics),
        "mean_portfolio_rating_distance": _mean_optional(
            value["portfolio_distance"] for value in task_metrics
        ),
        "mean_agent_calls": mean(value.cost_total.agent_calls for value in values),
        "mean_tool_calls": mean(value.cost_total.tool_calls for value in values),
        "mean_tokens": mean(
            value.cost_total.input_tokens + value.cost_total.output_tokens
            for value in values
        ),
        "mean_latency_ms": mean(value.cost_total.latency_ms for value in values),
        "same_as_static_path_rate": mean(
            value["same_static_path"] for value in task_metrics
        ),
        "unique_paths": len(paths),
    }


def evaluate_sets(
    static: Iterable[SchedulerTrajectory],
    **candidate_sets: Iterable[SchedulerTrajectory],
) -> dict[str, Any]:
    static_values = list(static)
    static_map = _index_by_task(static_values, name="static")
    results = {"static": evaluate_trajectories(static_values, static_map)}
    for name, trajectories in candidate_sets.items():
        values = list(trajectories)
        candidate_map = _index_by_task(values, name=name)
        missing = set(static_map) - set(candidate_map)
        unexpected = set(candidate_map) - set(static_map)
        if missing or unexpected:
            raise ValueError(
                f"{name} task set is not aligned with static: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        for key, candidate in candidate_map.items():
            _validate_comparable(static_map[key], candidate, name=name)
        results[name] = evaluate_trajectories(values, static_map)
    return {
        "evaluation_schema_version": "scheduler-evaluation-v1",
        "task_alignment": {
            "key_fields": ["task_id", "data_snapshot_id"],
            "task_count": len(static_map),
            "strict": True,
        },
        "modes": results,
    }


def _index_by_task(
    trajectories: Iterable[SchedulerTrajectory], *, name: str
) -> dict[tuple[str, str | None], SchedulerTrajectory]:
    indexed = {}
    for trajectory in trajectories:
        key = (trajectory.task_id, trajectory.data_snapshot_id)
        if key in indexed:
            raise ValueError(f"{name} contains duplicate trajectory for task {key}")
        indexed[key] = trajectory
    if not indexed:
        raise ValueError(f"{name} evaluation set is empty")
    return indexed


def _validate_comparable(
    static: SchedulerTrajectory,
    candidate: SchedulerTrajectory,
    *,
    name: str,
) -> None:
    for field in (
        "task_dataset_version",
        "information_cutoff",
        "expert_config_hash",
        "scheduler_max_steps",
    ):
        static_value = static.provenance.get(field)
        candidate_value = candidate.provenance.get(field)
        if static_value is None or candidate_value is None:
            raise ValueError(f"A/B provenance is missing {field} for {name}")
        if candidate_value != static_value:
            raise ValueError(f"A/B provenance mismatch for {name}: {field}")


def _trajectory_metrics(
    trajectory: SchedulerTrajectory,
    static: SchedulerTrajectory | None,
) -> dict[str, Any]:
    outputs = trajectory.final_outputs
    trader = parse_trader_action(outputs.get("trader_investment_plan"))
    rating = parse_portfolio_rating(outputs.get("final_trade_decision"))
    stop_count = sum(
        step.selected_action == SchedulerAction.STOP.value
        for step in trajectory.steps
    )
    legal = bool(trajectory.steps) and all(
        step.selected_action in step.valid_actions for step in trajectory.steps
    )
    metrics = {
        "completed": trajectory.execution_status == "completed",
        "failed": trajectory.execution_status != "completed",
        "legal": legal,
        "stop_correct": stop_count == 1
        and bool(trajectory.steps)
        and trajectory.steps[-1].selected_action == SchedulerAction.STOP.value,
        "trader_parse": trader is not None,
        "portfolio_parse": rating is not None,
        "fallback": trajectory.execution_status == "fallback",
        "budget_exhausted": trajectory.execution_status == "budget_exhausted",
        "context_overflow": trajectory.execution_status == "context_overflow",
        "trader_match": False,
        "portfolio_distance": None,
        "same_static_path": False,
    }
    if static is None:
        return metrics
    static_trader = parse_trader_action(
        static.final_outputs.get("trader_investment_plan")
    )
    static_rating = parse_portfolio_rating(
        static.final_outputs.get("final_trade_decision")
    )
    metrics["trader_match"] = trader is not None and trader == static_trader
    if rating in PORTFOLIO_RATINGS and static_rating in PORTFOLIO_RATINGS:
        metrics["portfolio_distance"] = abs(
            PORTFOLIO_RATINGS.index(rating) - PORTFOLIO_RATINGS.index(static_rating)
        )
    metrics["same_static_path"] = [
        step.selected_action for step in trajectory.steps
    ] == [step.selected_action for step in static.steps]
    return metrics


def _mean_optional(values: Iterable[int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--sft")
    parser.add_argument("--grpo")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidates = {}
    for name in ("teacher", "sft", "grpo"):
        source = getattr(args, name)
        if source:
            candidates[name] = TrajectoryStore(source).load()
    report = evaluate_sets(TrajectoryStore(args.static).load(), **candidates)
    write_json_atomic(Path(args.output), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
