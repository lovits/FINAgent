"""Build balanced, point-in-time task seeds for Static LangGraph collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

SEED_FAMILIES = (
    "earnings_window",
    "positive_momentum",
    "negative_momentum",
    "high_volatility",
    "volume_shock",
    "quiet_control",
)
DEFAULT_TICKERS = (
    "AAPL",
    "AMZN",
    "AMD",
    "GOOGL",
    "JNJ",
    "JPM",
    "MSFT",
    "NVDA",
    "TSLA",
    "WMT",
    "XOM",
)


def compute_feature_rows(
    price_frames: Mapping[str, pd.DataFrame],
    *,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """Compute features using only observations at or before each candidate date."""

    rows: list[dict[str, Any]] = []
    for ticker, source in sorted(price_frames.items()):
        if source.empty or "Volume" not in source:
            raise ValueError(f"missing OHLCV data for {ticker}")
        frame = source.copy().sort_index()
        index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        frame.index = index.normalize()

        close_column = "Adj Close" if "Adj Close" in frame else "Close"
        if close_column not in frame:
            raise ValueError(f"missing Close/Adj Close for {ticker}")
        close = pd.to_numeric(frame[close_column], errors="coerce")
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        daily_return = close.pct_change(fill_method=None)
        features = pd.DataFrame(
            {
                "trailing_return_5d": close / close.shift(5) - 1,
                "annualized_volatility_20d": daily_return.rolling(20).std()
                * math.sqrt(252),
                "volume_zscore_20d": (
                    (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
                ),
            },
            index=frame.index,
        ).dropna()

        for timestamp, values in features.iterrows():
            trade_date = timestamp.date()
            if start <= trade_date <= end:
                rows.append(
                    {
                        "ticker": ticker.upper(),
                        "trade_date": trade_date,
                        "trailing_return_5d": float(values["trailing_return_5d"]),
                        "annualized_volatility_20d": float(
                            values["annualized_volatility_20d"]
                        ),
                        "volume_zscore_20d": float(values["volume_zscore_20d"]),
                    }
                )
    if not rows:
        raise ValueError("no complete feature rows in the requested market window")
    return rows


def _rank_candidates(
    family: str,
    rows: Sequence[dict[str, Any]],
    earnings_dates: Mapping[str, set[date]],
) -> list[dict[str, Any]]:
    if family == "earnings_window":
        candidates = [
            row
            for row in rows
            if row["trade_date"] in earnings_dates.get(row["ticker"], set())
        ]
        return sorted(candidates, key=lambda row: (-row["trade_date"].toordinal(), row["ticker"]))
    if family == "positive_momentum":
        candidates = [row for row in rows if row["trailing_return_5d"] > 0]
    elif family == "negative_momentum":
        candidates = [row for row in rows if row["trailing_return_5d"] < 0]
    elif family == "high_volatility":
        candidates = list(rows)
    elif family == "volume_shock":
        candidates = [row for row in rows if row["volume_zscore_20d"] > 0]
    elif family == "quiet_control":
        candidates = list(rows)
    else:
        raise ValueError(f"unknown seed family: {family}")

    def score(row: dict[str, Any]) -> float:
        if family == "positive_momentum":
            return -row["trailing_return_5d"]
        if family == "negative_momentum":
            return row["trailing_return_5d"]
        if family == "high_volatility":
            return -row["annualized_volatility_20d"]
        if family == "volume_shock":
            return -row["volume_zscore_20d"]
        return (
            abs(row["trailing_return_5d"])
            + row["annualized_volatility_20d"]
            + 0.05 * abs(row["volume_zscore_20d"])
        )

    return sorted(
        candidates,
        key=lambda row: (score(row), row["ticker"], row["trade_date"]),
    )


def select_task_seeds(
    feature_rows: Sequence[dict[str, Any]],
    earnings_dates: Mapping[str, set[date]],
    *,
    tasks_per_family: int = 3,
    max_tasks_per_ticker: int = 2,
    min_gap_days: int = 21,
    dataset_split: str = "pilot",
    sector_by_ticker: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Select six balanced strata without consulting any future outcome."""

    selected: list[dict[str, Any]] = []
    ticker_dates: dict[str, list[date]] = defaultdict(list)
    used_pairs: set[tuple[str, date]] = set()

    for family in SEED_FAMILIES:
        family_tickers: set[str] = set()
        family_rows = []
        for row in _rank_candidates(family, feature_rows, earnings_dates):
            ticker = row["ticker"]
            trade_date = row["trade_date"]
            if ticker in family_tickers or (ticker, trade_date) in used_pairs:
                continue
            if len(ticker_dates[ticker]) >= max_tasks_per_ticker:
                continue
            if any(abs((trade_date - prior).days) < min_gap_days for prior in ticker_dates[ticker]):
                continue
            family_rows.append(row)
            family_tickers.add(ticker)
            ticker_dates[ticker].append(trade_date)
            used_pairs.add((ticker, trade_date))
            if len(family_rows) == tasks_per_family:
                break
        if len(family_rows) != tasks_per_family:
            raise ValueError(
                f"only {len(family_rows)} eligible rows for {family}; "
                "widen the window/universe or relax a declared constraint"
            )

        for row in family_rows:
            task = {
                "task_id": f"{row['ticker']}_{row['trade_date'].isoformat()}_{family}",
                "ticker": row["ticker"],
                "trade_date": row["trade_date"].isoformat(),
                "asset_type": "stock",
                "split": dataset_split,
                "seed_family": family,
                "selection_source": (
                    "yfinance_earnings_calendar"
                    if family == "earnings_window"
                    else "yfinance_ohlcv"
                ),
                "selection_features": {
                    "trailing_return_5d": round(row["trailing_return_5d"], 4),
                    "annualized_volatility_20d": round(
                        row["annualized_volatility_20d"], 4
                    ),
                    "volume_zscore_20d": round(row["volume_zscore_20d"], 2),
                },
                "status": "task_seed",
            }
            if sector_by_ticker is not None:
                task["sector"] = sector_by_ticker[row["ticker"]]
            selected.append(task)
    return selected


def build_manifest(
    tasks: Sequence[dict[str, Any]],
    *,
    start: date,
    end: date,
    max_tasks_per_ticker: int,
    min_gap_days: int,
    retrieved_at: str,
    dataset_id: str = "tradingagents-static-langgraph-pilot-v1",
    usage_boundary: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "status": "task_seed_only",
        "created_at": retrieved_at[:10],
        "market_window": [start.isoformat(), end.isoformat()],
        "task_count": len(tasks),
        "asset_type": "stock",
        "design_basis": {
            "tradingagents": "point-in-time daily multi-modal simulation",
            "fintradebench_2026": "calibration-then-scaling and numerical audit",
            "finance_agent_benchmark": "typed tasks plus quality-and-cost evaluation",
            "risk_first_agent_evaluation_2026": "tool and multi-step failure records",
        },
        "seed_families": dict(Counter(task["seed_family"] for task in tasks)),
        "selection_constraints": {
            "max_tasks_per_ticker": max_tasks_per_ticker,
            "min_calendar_days_between_same_ticker_tasks": min_gap_days,
            "price_features_use_information_through_trade_date": True,
            "future_returns_used_for_selection": False,
            "latest_data_cutoff": end.isoformat(),
        },
        "selection_features": {
            "trailing_return_5d": "Close[t] / Close[t-5] - 1",
            "annualized_volatility_20d": "std(daily_return[t-19:t]) * sqrt(252)",
            "volume_zscore_20d": "(Volume[t] - rolling_mean_20) / rolling_std_20",
        },
        "selection_method": {
            "version": "v1",
            "implementation": "training.scheduler.build_task_seeds",
            "family_order": list(SEED_FAMILIES),
            "quiet_control_score": "abs(return_5d) + volatility_20d + 0.05 * abs(volume_zscore_20d)",
            "silently_relax_constraints": False,
        },
        "source_snapshot": {
            "provider": "yfinance",
            "retrieved_at_utc": retrieved_at,
            "package_version": version("yfinance"),
            "raw_market_data_redistributed": False,
        },
        "post_run_labels": [
            "completed",
            "evidence_conflict",
            "data_sparse",
            "high_risk",
        ],
        "trajectory_generation": {
            "policy": "static",
            "capture_mode": "original_compiled_static_graph_event_stream",
            "trajectories_per_task": 1,
            "generator": "python -m training.scheduler.generate_data",
            "requires_fresh_model_api_credentials": True,
            "completed": False,
        },
        "usage_boundary": usage_boundary
        or (
            "Calibration pilot only. Review temporal integrity and trajectories before "
            "using any split as a held-out benchmark."
        ),
    }


def fetch_market_inputs(
    tickers: Sequence[str], start: date, end: date
) -> tuple[dict[str, pd.DataFrame], dict[str, set[date]]]:
    import yfinance as yf

    history_start = start - timedelta(days=60)
    exclusive_end = end + timedelta(days=1)
    prices: dict[str, pd.DataFrame] = {}
    earnings: dict[str, set[date]] = {}
    for ticker in tickers:  # sequential fetch avoids yfinance cache-lock races
        client = yf.Ticker(ticker)
        frame = client.history(
            start=history_start.isoformat(),
            end=exclusive_end.isoformat(),
            auto_adjust=False,
            actions=False,
        )
        if frame.empty:
            raise ValueError(f"yfinance returned no market data for {ticker}")
        prices[ticker] = frame
        calendar = client.get_earnings_dates(limit=24)
        dates: set[date] = set()
        if calendar is not None:
            for value in calendar.index:
                event_date = pd.Timestamp(value).date()
                if start <= event_date <= end:
                    dates.add(event_date)
        earnings[ticker] = dates
    return prices, earnings


def write_task_dataset(
    tasks: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    task_path: str | Path,
    manifest_path: str | Path,
) -> None:
    task_destination = Path(task_path)
    manifest_destination = Path(manifest_path)
    task_destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    task_content = "".join(
        json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n"
        for task in tasks
    )
    task_destination.write_text(task_content, encoding="utf-8")
    manifest_payload = dict(manifest)
    manifest_payload["files"] = {
        "tasks": {
            "name": task_destination.name,
            "format": "jsonl",
            "records": len(tasks),
            "sha256": hashlib.sha256(task_content.encode("utf-8")).hexdigest(),
        }
    }
    manifest_destination.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--tasks-per-family", type=int, default=3)
    parser.add_argument("--max-tasks-per-ticker", type=int, default=2)
    parser.add_argument("--min-gap-days", type=int, default=21)
    parser.add_argument("--output-tasks", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start > end:
        raise ValueError("start must not be after end")
    tickers = tuple(value.strip().upper() for value in args.tickers.split(",") if value.strip())
    prices, earnings = fetch_market_inputs(tickers, start, end)
    feature_rows = compute_feature_rows(prices, start=start, end=end)
    tasks = select_task_seeds(
        feature_rows,
        earnings,
        tasks_per_family=args.tasks_per_family,
        max_tasks_per_ticker=args.max_tasks_per_ticker,
        min_gap_days=args.min_gap_days,
    )
    retrieved_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest = build_manifest(
        tasks,
        start=start,
        end=end,
        max_tasks_per_ticker=args.max_tasks_per_ticker,
        min_gap_days=args.min_gap_days,
        retrieved_at=retrieved_at,
    )
    write_task_dataset(
        tasks,
        manifest,
        task_path=args.output_tasks,
        manifest_path=args.output_manifest,
    )
    print(json.dumps({"tasks": len(tasks), "tickers": len({t['ticker'] for t in tasks})}))


if __name__ == "__main__":
    main()
