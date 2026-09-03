"""Generate verified Static-Teacher or Strong-Teacher scheduler trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.scheduler.policy import StaticSchedulerPolicy
from tradingagents.scheduler.teacher_gateway import OpenRouterTeacherGateway
from tradingagents.scheduler.teacher_policy import StrongTeacherPolicy
from tradingagents.scheduler.trajectory_store import TrajectoryStore
from tradingagents.scheduler.verifier import verify_trajectory

from .environment import TradingAgentsRolloutEnvironment
from .reward import parse_trader_action, portfolio_similarity, score_trajectory


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
                for field in ("ticker", "trade_date"):
                    if not task.get(field):
                        raise ValueError(f"missing {field}")
                task.setdefault(
                    "task_id",
                    f"{task['ticker']}:{task['trade_date']}:{task.get('asset_type', 'stock')}",
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid task at {path}:{line_number}: {exc}") from exc
            tasks.append(task)
    if not tasks:
        raise ValueError(f"task file is empty: {path}")
    return tasks


def _quality_ok(
    final_state: dict[str, Any],
    static_state: dict[str, Any],
    *,
    min_rating_similarity: float,
) -> bool:
    if portfolio_similarity(
        str(final_state.get("final_trade_decision") or ""),
        str(static_state.get("final_trade_decision") or ""),
    ) < min_rating_similarity:
        return False
    candidate = parse_trader_action(final_state.get("trader_investment_plan"))
    reference = parse_trader_action(static_state.get("trader_investment_plan"))
    return candidate is not None and candidate == reference


def generate(
    *,
    tasks: list[dict[str, Any]],
    output_dir: str | Path,
    policy_kind: str,
    trajectories_per_task: int,
    selected_analysts: tuple[str, ...],
    min_rating_similarity: float,
    config: dict[str, Any] | None = None,
) -> dict[str, int]:
    if policy_kind == "static":
        policy = StaticSchedulerPolicy(
            max_debate_rounds=(config or DEFAULT_CONFIG)["max_debate_rounds"],
            max_risk_rounds=(config or DEFAULT_CONFIG)["max_risk_discuss_rounds"],
        )
        source = "static_teacher"
    elif policy_kind == "teacher":
        policy = StrongTeacherPolicy(
            OpenRouterTeacherGateway(model=(config or DEFAULT_CONFIG)["teacher_model"])
        )
        source = "strong_teacher_verified"
    else:
        raise ValueError("policy_kind must be 'static' or 'teacher'")

    output = Path(output_dir)
    accepted_store = TrajectoryStore(output / "accepted.jsonl")
    rejected_store = TrajectoryStore(output / "rejected.jsonl")
    environment = TradingAgentsRolloutEnvironment(
        config or DEFAULT_CONFIG,
        selected_analysts=selected_analysts,
    )
    counts = {"accepted": 0, "rejected": 0}
    for task_index, task in enumerate(tasks):
        for sample_index in range(trajectories_per_task):
            result = environment.run(
                task,
                policy,
                seed=task_index * trajectories_per_task + sample_index,
            )
            quality_ok = policy_kind == "static" or _quality_ok(
                result.final_state,
                result.static_state,
                min_rating_similarity=min_rating_similarity,
            )
            verification = verify_trajectory(
                result.trajectory,
                result.final_state,
                quality_ok=quality_ok,
            )
            trajectory = result.trajectory
            trajectory.metadata["source"] = source
            trajectory.verifier_version = verification.verifier_version
            trajectory.reward = score_trajectory(
                trajectory, result.final_state, result.static_state
            ).to_dict()
            if verification.accepted:
                trajectory.status = "accepted"
                trajectory.failure_reason = None
                accepted_store.append(trajectory)
                counts["accepted"] += 1
            else:
                trajectory.status = "rejected"
                trajectory.failure_reason = verification.reason
                rejected_store.append(trajectory)
                counts["rejected"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=("static", "teacher"), required=True)
    parser.add_argument("--trajectories-per-task", type=int, default=None)
    parser.add_argument(
        "--analysts", default="market,social,news,fundamentals"
    )
    parser.add_argument("--min-rating-similarity", type=float, default=0.5)
    args = parser.parse_args()
    trajectories = args.trajectories_per_task or (1 if args.policy == "static" else 2)
    counts = generate(
        tasks=load_tasks(args.tasks),
        output_dir=args.output_dir,
        policy_kind=args.policy,
        trajectories_per_task=trajectories,
        selected_analysts=tuple(part.strip() for part in args.analysts.split(",") if part.strip()),
        min_rating_similarity=args.min_rating_similarity,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
