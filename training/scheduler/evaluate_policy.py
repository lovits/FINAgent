"""Static/SFT/RL scheduler evaluation records and aggregate reports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.scheduler.trajectory import SchedulerTrajectory
from tradingagents.scheduler.trajectory_store import TrajectoryStore

from .reward import parse_trader_action


@dataclass(frozen=True)
class EvaluationRecord:
    task_id: str
    mode: str
    policy_id: str
    completed: bool
    invalid_actions: int
    fallback: bool
    trader_action: str | None
    portfolio_rating: str | None
    agent_calls: int
    tool_calls: int
    total_tokens: int
    latency_ms: float
    path_length: int
    repeated_actions: int


def evaluate_trajectory(
    trajectory: SchedulerTrajectory,
    final_state: dict[str, Any],
    *,
    mode_label: str | None = None,
) -> EvaluationRecord:
    actions = [step.selected_action for step in trajectory.steps]
    completed = all(
        isinstance(final_state.get(field), str) and bool(final_state[field].strip())
        for field in ("investment_plan", "trader_investment_plan", "final_trade_decision")
    )
    decision = str(final_state.get("final_trade_decision") or "")
    return EvaluationRecord(
        task_id=trajectory.task_id,
        mode=mode_label or trajectory.mode,
        policy_id=trajectory.policy_id,
        completed=completed,
        invalid_actions=sum(step.error is not None for step in trajectory.steps),
        fallback=trajectory.fallback_reason is not None,
        trader_action=parse_trader_action(final_state.get("trader_investment_plan")),
        portfolio_rating=parse_rating(decision) if decision else None,
        agent_calls=sum(step.agent_node is not None for step in trajectory.steps),
        tool_calls=sum(step.tool_calls for step in trajectory.steps),
        total_tokens=sum(step.input_tokens + step.output_tokens for step in trajectory.steps),
        latency_ms=sum(step.latency_ms for step in trajectory.steps),
        path_length=len(actions),
        repeated_actions=len(actions) - len(set(actions)),
    )


def summarize(records: Iterable[EvaluationRecord]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        groups[record.mode].append(record)
    return {
        mode: {
            "tasks": float(len(items)),
            "completion_rate": mean(item.completed for item in items),
            "invalid_action_rate": mean(item.invalid_actions > 0 for item in items),
            "fallback_rate": mean(item.fallback for item in items),
            "mean_agent_calls": mean(item.agent_calls for item in items),
            "mean_tool_calls": mean(item.tool_calls for item in items),
            "mean_total_tokens": mean(item.total_tokens for item in items),
            "mean_latency_ms": mean(item.latency_ms for item in items),
            "mean_path_length": mean(item.path_length for item in items),
            "mean_repeated_actions": mean(item.repeated_actions for item in items),
        }
        for mode, items in groups.items()
    }


def write_evaluation_report(
    records: Iterable[EvaluationRecord], path: str | Path
) -> None:
    records = list(records)
    payload = {
        "summary": summarize(records),
        "records": [asdict(record) for record in records],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Mode-labelled trajectory file, for example static=path.jsonl",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = []
    for item in args.input:
        if "=" not in item:
            raise ValueError("--input must use mode=path format")
        mode, path = item.split("=", 1)
        for trajectory in TrajectoryStore(path).load():
            final_state = {
                "investment_plan": trajectory.investment_plan,
                "trader_investment_plan": trajectory.trader_result,
                "final_trade_decision": trajectory.final_decision,
            }
            records.append(
                evaluate_trajectory(trajectory, final_state, mode_label=mode)
            )
    write_evaluation_report(records, args.output)


if __name__ == "__main__":
    main()
