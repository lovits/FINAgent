"""Build a split-aware, cross-sector scheduler task pool from a frozen plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .build_task_seeds import (
    SEED_FAMILIES,
    build_manifest,
    compute_feature_rows,
    fetch_market_inputs,
    select_task_seeds,
    write_task_dataset,
)


def load_dataset_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_plan(plan)
    return plan


def _validate_plan(plan: Mapping[str, Any]) -> None:
    for field in (
        "dataset_id",
        "market_window",
        "tasks_per_family_by_split",
        "max_tasks_per_ticker",
        "min_gap_days",
        "universe",
    ):
        if not plan.get(field):
            raise ValueError(f"dataset plan is missing {field}")

    start, end = (date.fromisoformat(value) for value in plan["market_window"])
    if start > end:
        raise ValueError("dataset plan market_window is reversed")
    split_targets = plan["tasks_per_family_by_split"]
    if not all(isinstance(value, int) and value > 0 for value in split_targets.values()):
        raise ValueError("every split target must be a positive integer")

    ticker_records = plan["universe"]
    tickers = [record.get("ticker") for record in ticker_records]
    if len(tickers) != len(set(tickers)):
        raise ValueError("dataset plan tickers must be unique")
    for record in ticker_records:
        if not all(record.get(field) for field in ("ticker", "sector", "split")):
            raise ValueError("every universe row needs ticker, sector, and split")
        if record["ticker"] != record["ticker"].upper():
            raise ValueError(f"ticker must be uppercase: {record['ticker']}")
        if record["split"] not in split_targets:
            raise ValueError(f"unknown split for {record['ticker']}: {record['split']}")

    max_per_ticker = int(plan["max_tasks_per_ticker"])
    for split, target in split_targets.items():
        ticker_count = sum(record["split"] == split for record in ticker_records)
        if target > ticker_count:
            raise ValueError(
                f"{split} needs {target} distinct tickers per family, has {ticker_count}"
            )
        if target * len(SEED_FAMILIES) > ticker_count * max_per_ticker:
            raise ValueError(f"{split} has insufficient total ticker capacity")


def select_scaled_tasks(
    feature_rows: Sequence[dict[str, Any]],
    earnings_dates: Mapping[str, set[date]],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sectors = {record["ticker"]: record["sector"] for record in plan["universe"]}
    split_by_ticker = {record["ticker"]: record["split"] for record in plan["universe"]}
    selected = []
    for dataset_split, target in plan["tasks_per_family_by_split"].items():
        allowed = {
            ticker for ticker, split in split_by_ticker.items() if split == dataset_split
        }
        split_rows = [row for row in feature_rows if row["ticker"] in allowed]
        split_earnings = {
            ticker: dates for ticker, dates in earnings_dates.items() if ticker in allowed
        }
        selected.extend(
            select_task_seeds(
                split_rows,
                split_earnings,
                tasks_per_family=int(target),
                max_tasks_per_ticker=int(plan["max_tasks_per_ticker"]),
                min_gap_days=int(plan["min_gap_days"]),
                dataset_split=dataset_split,
                sector_by_ticker=sectors,
            )
        )
    return selected


def build_scaled_manifest(
    tasks: Sequence[dict[str, Any]],
    plan: Mapping[str, Any],
    *,
    retrieved_at: str,
) -> dict[str, Any]:
    start, end = (date.fromisoformat(value) for value in plan["market_window"])
    manifest = build_manifest(
        tasks,
        start=start,
        end=end,
        max_tasks_per_ticker=int(plan["max_tasks_per_ticker"]),
        min_gap_days=int(plan["min_gap_days"]),
        retrieved_at=retrieved_at,
        dataset_id=str(plan["dataset_id"]),
        usage_boundary=(
            "Expanded scheduler task pool. Generate trajectories in reviewed stages; "
            "keep the ticker-disjoint test split out of SFT and RL training."
        ),
    )
    universe = plan["universe"]
    manifest.update(
        {
            "dataset_kind": "expanded_scheduler_task_pool",
            "split_strategy": {
                "type": "ticker_disjoint",
                "assignment": "frozen_in_dataset_plan",
                "purpose": "prevent the same company from crossing train/validation/test",
            },
            "tasks_per_family_by_split": dict(plan["tasks_per_family_by_split"]),
            "splits": {
                split: {
                    "task_count": sum(task["split"] == split for task in tasks),
                    "ticker_count": len(
                        {task["ticker"] for task in tasks if task["split"] == split}
                    ),
                }
                for split in plan["tasks_per_family_by_split"]
            },
            "candidate_universe": {
                "ticker_count": len(universe),
                "sector_count": len({record["sector"] for record in universe}),
                "sector_counts": dict(Counter(record["sector"] for record in universe)),
                "tickers_by_split": {
                    split: [
                        record["ticker"] for record in universe if record["split"] == split
                    ]
                    for split in plan["tasks_per_family_by_split"]
                },
            },
        }
    )
    manifest["trajectory_generation"].update(
        {
            "recommended_stages": [18, 60, len(tasks)],
            "completed": False,
        }
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-tasks", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    plan = load_dataset_plan(args.plan)
    start, end = (date.fromisoformat(value) for value in plan["market_window"])
    tickers = tuple(record["ticker"] for record in plan["universe"])
    prices, earnings = fetch_market_inputs(tickers, start, end)
    feature_rows = compute_feature_rows(prices, start=start, end=end)
    tasks = select_scaled_tasks(feature_rows, earnings, plan)
    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest = build_scaled_manifest(tasks, plan, retrieved_at=retrieved_at)
    write_task_dataset(
        tasks,
        manifest,
        task_path=args.output_tasks,
        manifest_path=args.output_manifest,
    )
    print(
        json.dumps(
            {
                "dataset_id": plan["dataset_id"],
                "tasks": len(tasks),
                "tickers": len({task["ticker"] for task in tasks}),
                "splits": manifest["splits"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
