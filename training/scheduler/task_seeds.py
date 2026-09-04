"""Build a balanced, leakage-aware scheduler task pool from historical features."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from tradingagents.scheduler.store import write_json_atomic

from .profile import (
    CHINESE_OUTPUT_LANGUAGE,
    SHALLOW_RESEARCH_DEPTH,
    validate_training_profile,
)

TASK_DATASET_VERSION = "scheduler-tasks-v2"
SEED_FAMILIES = (
    "earnings_window",
    "positive_momentum",
    "negative_momentum",
    "high_volatility",
    "volume_shock",
    "quiet_control",
)
TaskSplit = Literal["train", "validation", "reserve"]

ANALYST_INPUT_SETS = (
    ("market",),
    ("market", "news"),
    ("market", "social", "news"),
    ("fundamentals",),
    ("social",),
    ("market", "social"),
    ("market", "news", "fundamentals"),
    ("news",),
    ("market", "fundamentals"),
    ("social", "news", "fundamentals"),
    ("market", "social", "news", "fundamentals"),
    ("social", "news"),
    ("market", "social", "fundamentals"),
    ("news", "fundamentals"),
    ("social", "fundamentals"),
)


@dataclass(frozen=True)
class TaskSeed:
    task_id: str
    ticker: str
    trade_date: str
    asset_type: str
    split: TaskSplit
    seed_family: str
    sector: str
    data_snapshot_id: str
    information_cutoff: str
    selection_features: dict[str, float | bool]
    selected_analysts: tuple[str, ...]
    research_depth: Literal["shallow"]
    output_language: Literal["Chinese"]
    dataset_version: str = TASK_DATASET_VERSION

    def __post_init__(self) -> None:
        if self.seed_family not in SEED_FAMILIES:
            raise ValueError(f"unknown seed family: {self.seed_family}")
        if self.split not in {"train", "validation", "reserve"}:
            raise ValueError(f"unknown task split: {self.split}")
        date.fromisoformat(self.trade_date)
        if self.asset_type != "stock":
            raise ValueError("scheduler task v2 supports stock tasks only")
        validate_training_profile(self.selected_analysts)
        if self.research_depth != SHALLOW_RESEARCH_DEPTH:
            raise ValueError("scheduler task v2 only supports shallow research")
        if self.output_language != CHINESE_OUTPUT_LANGUAGE:
            raise ValueError("scheduler task v2 only supports Chinese output")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_feature_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                _validate_feature_row(row)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid feature row at {source}:{line_number}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"feature file is empty: {source}")
    return rows


def _validate_feature_row(row: Mapping[str, Any]) -> None:
    for field in (
        "ticker",
        "trade_date",
        "sector",
        "data_snapshot_id",
        "information_cutoff",
        "trailing_return_5d",
        "annualized_volatility_20d",
        "volume_zscore_20d",
        "earnings_window",
    ):
        if field not in row:
            raise KeyError(field)
    date.fromisoformat(str(row["trade_date"]))
    for field in (
        "trailing_return_5d",
        "annualized_volatility_20d",
        "volume_zscore_20d",
    ):
        float(row[field])


def _eligible(family: str, row: Mapping[str, Any]) -> bool:
    if family == "earnings_window":
        return bool(row["earnings_window"])
    if family == "positive_momentum":
        return float(row["trailing_return_5d"]) > 0
    if family == "negative_momentum":
        return float(row["trailing_return_5d"]) < 0
    if family == "volume_shock":
        return float(row["volume_zscore_20d"]) > 0
    return True


def _priority(family: str, row: Mapping[str, Any]) -> tuple[float, str, str]:
    trailing = float(row["trailing_return_5d"])
    volatility = float(row["annualized_volatility_20d"])
    volume = float(row["volume_zscore_20d"])
    if family == "earnings_window":
        score = -date.fromisoformat(str(row["trade_date"])).toordinal()
    elif family == "positive_momentum":
        score = -trailing
    elif family == "negative_momentum":
        score = trailing
    elif family == "high_volatility":
        score = -volatility
    elif family == "volume_shock":
        score = -volume
    else:
        score = abs(trailing) + volatility + 0.05 * abs(volume)
    return score, str(row["ticker"]), str(row["trade_date"])


def select_balanced_task_pool(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    tasks_per_family: int = 50,
    train_per_family: int = 10,
    validation_per_family: int = 2,
    max_tasks_per_ticker: int = 10,
    min_gap_days: int = 21,
) -> list[TaskSeed]:
    if train_per_family + validation_per_family > tasks_per_family:
        raise ValueError("train and validation quotas exceed tasks_per_family")
    selected: list[TaskSeed] = []
    used_pairs: set[tuple[str, str]] = set()
    ticker_dates: dict[str, list[date]] = defaultdict(list)
    ticker_counts: Counter[str] = Counter()

    for family in SEED_FAMILIES:
        candidates = sorted(
            (row for row in feature_rows if _eligible(family, row)),
            key=lambda row: _priority(family, row),
        )
        family_rows = _select_family_rows(
            candidates,
            tasks_per_family=tasks_per_family,
            max_tasks_per_ticker=max_tasks_per_ticker,
            min_gap_days=min_gap_days,
            used_pairs=used_pairs,
            ticker_dates=ticker_dates,
            ticker_counts=ticker_counts,
        )
        if len(family_rows) != tasks_per_family:
            raise ValueError(
                f"only {len(family_rows)} eligible unique rows for {family}; "
                f"required {tasks_per_family}"
            )
        task_offset = len(selected)
        selected.extend(
            _task_from_row(
                row,
                family,
                _split_for_index(
                    index,
                    train_per_family=train_per_family,
                    validation_per_family=validation_per_family,
                ),
                analyst_set_index=task_offset + index,
            )
            for index, row in enumerate(family_rows)
        )
    return selected


def _select_family_rows(
    candidates: Iterable[Mapping[str, Any]],
    *,
    tasks_per_family: int,
    max_tasks_per_ticker: int,
    min_gap_days: int,
    used_pairs: set[tuple[str, str]],
    ticker_dates: dict[str, list[date]],
    ticker_counts: Counter[str],
) -> list[Mapping[str, Any]]:
    rows = []
    family_tickers: set[str] = set()
    for row in candidates:
        ticker = str(row["ticker"]).upper()
        trade_date = date.fromisoformat(str(row["trade_date"]))
        pair = (ticker, trade_date.isoformat())
        if pair in used_pairs or ticker in family_tickers:
            continue
        if ticker_counts[ticker] >= max_tasks_per_ticker:
            continue
        if any(abs((trade_date - prior).days) < min_gap_days for prior in ticker_dates[ticker]):
            continue
        rows.append(row)
        used_pairs.add(pair)
        family_tickers.add(ticker)
        ticker_dates[ticker].append(trade_date)
        ticker_counts[ticker] += 1
        if len(rows) == tasks_per_family:
            break
    return rows


def _split_for_index(
    index: int, *, train_per_family: int, validation_per_family: int
) -> TaskSplit:
    if index < train_per_family:
        return "train"
    if index < train_per_family + validation_per_family:
        return "validation"
    return "reserve"


def _task_from_row(
    row: Mapping[str, Any],
    family: str,
    split: TaskSplit,
    *,
    analyst_set_index: int,
) -> TaskSeed:
    ticker = str(row["ticker"]).upper()
    trade_date = str(row["trade_date"])
    return TaskSeed(
        task_id=f"{ticker}_{trade_date}_{family}",
        ticker=ticker,
        trade_date=trade_date,
        asset_type="stock",
        split=split,
        seed_family=family,
        sector=str(row["sector"]),
        data_snapshot_id=str(row["data_snapshot_id"]),
        information_cutoff=str(row["information_cutoff"]),
        selection_features={
            "trailing_return_5d": round(float(row["trailing_return_5d"]), 6),
            "annualized_volatility_20d": round(
                float(row["annualized_volatility_20d"]), 6
            ),
            "volume_zscore_20d": round(float(row["volume_zscore_20d"]), 6),
            "earnings_window": bool(row["earnings_window"]),
        },
        selected_analysts=ANALYST_INPUT_SETS[
            analyst_set_index % len(ANALYST_INPUT_SETS)
        ],
        research_depth=SHALLOW_RESEARCH_DEPTH,
        output_language=CHINESE_OUTPUT_LANGUAGE,
    )


def write_task_dataset(tasks: Sequence[TaskSeed], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[TaskSeed]] = defaultdict(list)
    for task in tasks:
        grouped[task.split].append(task)
    _write_jsonl(destination / "pool_300.jsonl", tasks)
    _write_jsonl(destination / "train_60.jsonl", grouped["train"])
    _write_jsonl(destination / "validation_12.jsonl", grouped["validation"])
    manifest = {
        "dataset_version": TASK_DATASET_VERSION,
        "total": len(tasks),
        "split_counts": {key: len(values) for key, values in grouped.items()},
        "family_counts": dict(Counter(task.seed_family for task in tasks)),
        "analyst_count_counts": dict(
            sorted(Counter(len(task.selected_analysts) for task in tasks).items())
        ),
        "analyst_set_counts": dict(
            sorted(Counter("+".join(task.selected_analysts) for task in tasks).items())
        ),
        "research_depth_counts": dict(Counter(task.research_depth for task in tasks)),
        "output_language_counts": dict(Counter(task.output_language for task in tasks)),
    }
    write_json_atomic(destination / "manifest.json", manifest)


def _write_jsonl(path: Path, tasks: Iterable[TaskSeed]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    tasks = select_balanced_task_pool(load_feature_rows(args.features))
    write_task_dataset(tasks, args.output_dir)
    print(json.dumps({"tasks": len(tasks), "output_dir": args.output_dir}))


if __name__ == "__main__":
    main()
