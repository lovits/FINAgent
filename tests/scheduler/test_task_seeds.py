import json
from collections import Counter

import pytest

from training.scheduler.task_seeds import (
    ANALYST_INPUT_SETS,
    SEED_FAMILIES,
    load_feature_rows,
    select_balanced_task_pool,
    write_task_dataset,
)


def _row(prefix: str, index: int, **overrides) -> dict:
    row = {
        "ticker": f"{prefix}{index:03d}",
        "trade_date": "2026-01-05",
        "sector": "Technology",
        "data_snapshot_id": "snapshot-2026-01-05",
        "information_cutoff": "2026-01-05T23:59:59Z",
        "trailing_return_5d": 0.0,
        "annualized_volatility_20d": 0.1,
        "volume_zscore_20d": 0.0,
        "earnings_window": False,
    }
    row.update(overrides)
    return row


def _features() -> list[dict]:
    rows = []
    for index in range(50):
        rows.append(_row("E", index, earnings_window=True))
        rows.append(_row("P", index, trailing_return_5d=1.0 + index / 1000))
        rows.append(_row("N", index, trailing_return_5d=-1.0 - index / 1000))
        rows.append(_row("H", index, annualized_volatility_20d=3.0 + index / 1000))
        rows.append(_row("V", index, volume_zscore_20d=10.0 + index / 1000))
        rows.append(_row("Q", index))
    return rows


def test_builds_exact_balanced_300_task_pool() -> None:
    tasks = select_balanced_task_pool(_features())

    assert len(tasks) == 300
    assert Counter(task.seed_family for task in tasks) == dict.fromkeys(SEED_FAMILIES, 50)
    assert Counter(task.split for task in tasks) == {
        "train": 60,
        "validation": 12,
        "reserve": 228,
    }
    assert len({(task.ticker, task.trade_date) for task in tasks}) == 300
    assert {task.research_depth for task in tasks} == {"shallow"}
    assert {task.output_language for task in tasks} == {"Chinese"}
    assert Counter(task.selected_analysts for task in tasks) == dict.fromkeys(
        ANALYST_INPUT_SETS, 20
    )


def test_writes_pool_splits_and_manifest(tmp_path) -> None:
    tasks = select_balanced_task_pool(_features())
    write_task_dataset(tasks, tmp_path)

    assert len((tmp_path / "pool_300.jsonl").read_text().splitlines()) == 300
    assert len((tmp_path / "train_60.jsonl").read_text().splitlines()) == 60
    assert len((tmp_path / "validation_12.jsonl").read_text().splitlines()) == 12
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["split_counts"] == {
        "reserve": 228,
        "train": 60,
        "validation": 12,
    }
    assert manifest["analyst_count_counts"] == {
        "1": 80,
        "2": 120,
        "3": 80,
        "4": 20,
    }
    assert manifest["research_depth_counts"] == {"shallow": 300}
    assert manifest["output_language_counts"] == {"Chinese": 300}


def test_feature_loader_rejects_missing_fields(tmp_path) -> None:
    source = tmp_path / "features.jsonl"
    source.write_text('{"ticker":"AAPL"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid feature row"):
        load_feature_rows(source)
