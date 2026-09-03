from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from training.scheduler.build_task_seeds import write_task_dataset
from training.scheduler.scaled_task_seeds import (
    build_scaled_manifest,
    load_dataset_plan,
    select_scaled_tasks,
)
from training.scheduler.validate_task_manifest import validate_task_manifest

PLAN_PATH = (
    Path(__file__).parents[2]
    / "training/scheduler/configs/datasets/static_langgraph_300_v1.json"
)


def test_checked_in_plan_defines_300_tasks_and_balanced_sector_splits():
    plan = load_dataset_plan(PLAN_PATH)
    assert sum(plan["tasks_per_family_by_split"].values()) * 6 == 300
    assert Counter(record["split"] for record in plan["universe"]) == {
        "train": 44,
        "validation": 11,
        "test": 11,
    }
    assert set(Counter(record["sector"] for record in plan["universe"]).values()) == {6}


def test_scaled_selection_is_balanced_and_ticker_disjoint(tmp_path):
    plan = {
        "dataset_id": "fixture-18",
        "market_window": ["2026-01-01", "2026-06-30"],
        "tasks_per_family_by_split": {"train": 1, "validation": 1, "test": 1},
        "max_tasks_per_ticker": 6,
        "min_gap_days": 21,
        "universe": [
            {"ticker": "AAA", "sector": "technology", "split": "train"},
            {"ticker": "BBB", "sector": "financials", "split": "validation"},
            {"ticker": "CCC", "sector": "health_care", "split": "test"},
        ],
    }
    dates = (
        date(2026, 1, 2),
        date(2026, 2, 3),
        date(2026, 3, 3),
        date(2026, 4, 3),
        date(2026, 5, 4),
        date(2026, 6, 4),
    )
    values = (
        (0.01, 0.20, 0.1),
        (0.40, 0.30, 0.1),
        (-0.40, 0.30, 0.1),
        (0.05, 1.20, 0.1),
        (0.05, 0.30, 6.0),
        (0.00, 0.05, 0.0),
    )
    rows = []
    earnings = {}
    for ticker in ("AAA", "BBB", "CCC"):
        earnings[ticker] = {dates[0]}
        for trade_date, metrics in zip(dates, values, strict=True):
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "trailing_return_5d": metrics[0],
                    "annualized_volatility_20d": metrics[1],
                    "volume_zscore_20d": metrics[2],
                }
            )

    tasks = select_scaled_tasks(rows, earnings, plan)
    assert len(tasks) == 18
    assert Counter(task["split"] for task in tasks) == {
        "train": 6,
        "validation": 6,
        "test": 6,
    }
    assert all(task.get("sector") for task in tasks)

    manifest = build_scaled_manifest(
        tasks,
        plan,
        retrieved_at="2026-09-03T00:00:00+00:00",
    )
    task_path = tmp_path / "tasks.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_task_dataset(
        tasks,
        manifest,
        task_path=task_path,
        manifest_path=manifest_path,
    )
    result = validate_task_manifest(task_path, manifest_path)
    assert result["splits"] == {"train": 6, "validation": 6, "test": 6}

    task_path.write_text(task_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        validate_task_manifest(task_path, manifest_path)
