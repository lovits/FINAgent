from datetime import date

import pandas as pd

from training.scheduler.build_task_seeds import (
    SEED_FAMILIES,
    build_manifest,
    compute_feature_rows,
    select_task_seeds,
    write_task_dataset,
)
from training.scheduler.validate_task_manifest import validate_task_manifest


def _prices():
    index = pd.bdate_range("2026-01-01", periods=50)
    return pd.DataFrame(
        {
            "Adj Close": [100 + index for index in range(50)],
            "Volume": [1_000 + 10 * index for index in range(50)],
        },
        index=index,
    )


def test_feature_at_a_date_does_not_change_when_future_prices_change():
    original = _prices()
    target = original.index[35]
    changed = original.copy()
    changed.loc[changed.index > target, "Adj Close"] *= 10
    changed.loc[changed.index > target, "Volume"] *= 10

    kwargs = {"start": target.date(), "end": target.date()}
    first = compute_feature_rows({"TEST": original}, **kwargs)[0]
    second = compute_feature_rows({"TEST": changed}, **kwargs)[0]
    assert first == second


def test_balanced_selection_writes_a_manifest_accepted_by_validator(tmp_path):
    rows = []
    earnings = {}
    families = {
        "earnings": (0.01, 0.20, 0.0),
        "positive": (0.30, 0.30, 0.2),
        "negative": (-0.30, 0.30, 0.2),
        "volatile": (0.01, 1.00, 0.2),
        "volume": (0.01, 0.20, 5.0),
        "quiet": (0.0, 0.05, 0.0),
    }
    prefixes = {
        "earnings": "EA",
        "positive": "PO",
        "negative": "NE",
        "volatile": "HV",
        "volume": "VS",
        "quiet": "QC",
    }
    day = 1
    for group, values in families.items():
        for index in range(3):
            ticker = f"{prefixes[group]}{index}"
            trade_date = date(2026, 6, day)
            day += 1
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "trailing_return_5d": values[0],
                    "annualized_volatility_20d": values[1],
                    "volume_zscore_20d": values[2],
                }
            )
            if group == "earnings":
                earnings[ticker] = {trade_date}

    tasks = select_task_seeds(rows, earnings)
    assert len(tasks) == 18
    assert {task["seed_family"] for task in tasks} == set(SEED_FAMILIES)
    assert all("future_return" not in task for task in tasks)

    manifest = build_manifest(
        tasks,
        start=date(2026, 6, 1),
        end=date(2026, 6, 30),
        max_tasks_per_ticker=2,
        min_gap_days=21,
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
    assert result["tasks"] == 18
    assert result["status"] == "valid_task_seed"
