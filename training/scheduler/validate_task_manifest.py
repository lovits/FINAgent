"""Validate scheduler task seeds before any paid LangGraph generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .generate_data import load_tasks

REQUIRED_FEATURES = {
    "trailing_return_5d",
    "annualized_volatility_20d",
    "volume_zscore_20d",
}
FORBIDDEN_SEED_FIELDS = {"future_return", "future_alpha", "target_action", "answer"}


def validate_task_manifest(
    task_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    tasks = load_tasks(task_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected_count = int(manifest["task_count"])
    if len(tasks) != expected_count:
        raise ValueError(f"expected {expected_count} tasks, found {len(tasks)}")
    task_file = (manifest.get("files") or {}).get("tasks")
    if task_file:
        if int(task_file["records"]) != len(tasks):
            raise ValueError("task file record count does not match the manifest")
        actual_hash = hashlib.sha256(Path(task_path).read_bytes()).hexdigest()
        if actual_hash != task_file["sha256"]:
            raise ValueError("task file checksum does not match the manifest")

    ids = [task["task_id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task_id values must be unique")
    family_counts = Counter(task.get("seed_family") for task in tasks)
    if dict(family_counts) != manifest["seed_families"]:
        raise ValueError("seed family counts do not match the manifest")
    split_counts = Counter(task.get("split") for task in tasks)
    ticker_splits: dict[str, set[str]] = defaultdict(set)

    start, end = (date.fromisoformat(value) for value in manifest["market_window"])
    constraints = manifest["selection_constraints"]
    per_ticker: dict[str, list[date]] = defaultdict(list)
    for task in tasks:
        task_date = date.fromisoformat(task["trade_date"])
        if not start <= task_date <= end:
            raise ValueError(f"task date is outside the market window: {task['task_id']}")
        if FORBIDDEN_SEED_FIELDS.intersection(task):
            raise ValueError(f"task seed contains a future/answer field: {task['task_id']}")
        features = task.get("selection_features")
        if not isinstance(features, dict) or set(features) != REQUIRED_FEATURES:
            raise ValueError(f"invalid selection features: {task['task_id']}")
        if not all(math.isfinite(float(value)) for value in features.values()):
            raise ValueError(f"non-finite selection feature: {task['task_id']}")
        per_ticker[task["ticker"]].append(task_date)
        ticker_splits[task["ticker"]].add(str(task.get("split")))

    expected_splits = manifest.get("splits")
    if expected_splits:
        expected_counts = {
            split: int(details["task_count"])
            for split, details in expected_splits.items()
        }
        if dict(split_counts) != expected_counts:
            raise ValueError("split task counts do not match the manifest")
        family_targets = manifest.get("tasks_per_family_by_split", {})
        actual_by_split_family = Counter(
            (task.get("split"), task.get("seed_family")) for task in tasks
        )
        for split, target in family_targets.items():
            for family in manifest["seed_families"]:
                if actual_by_split_family[(split, family)] != int(target):
                    raise ValueError(f"{split}/{family} count does not match the manifest")

    split_strategy = manifest.get("split_strategy") or {}
    if split_strategy.get("type") == "ticker_disjoint":
        overlap = {ticker: splits for ticker, splits in ticker_splits.items() if len(splits) > 1}
        if overlap:
            raise ValueError(f"ticker-disjoint split violation: {sorted(overlap)}")
        if any(not task.get("sector") for task in tasks):
            raise ValueError("expanded ticker-disjoint tasks require sector metadata")

    max_tasks = int(constraints["max_tasks_per_ticker"])
    min_gap = int(constraints["min_calendar_days_between_same_ticker_tasks"])
    for ticker, dates in per_ticker.items():
        if len(dates) > max_tasks:
            raise ValueError(f"{ticker} exceeds max_tasks_per_ticker")
        ordered = sorted(dates)
        if any(
            (right - left).days < min_gap
            for left, right in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError(f"{ticker} has tasks closer than {min_gap} days")

    return {
        "dataset_id": manifest["dataset_id"],
        "tasks": len(tasks),
        "tickers": len(per_ticker),
        "seed_families": dict(family_counts),
        "splits": dict(split_counts),
        "market_window": [start.isoformat(), end.isoformat()],
        "status": "valid_task_seed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_task_manifest(args.tasks, args.manifest),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
