"""Fetch historical OHLCV data and derive task-selection features without lookahead."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


def _snapshot_id(ticker: str, history: pd.DataFrame) -> str:
    selected = history.loc[:, ["Close", "Volume"]].sort_index()
    digest = hashlib.sha256(selected.to_csv().encode()).hexdigest()[:16]
    return f"yfinance:{ticker.upper()}:{digest}"


def compute_feature_rows(
    ticker: str,
    history: pd.DataFrame,
    *,
    sector: str = "unknown",
    earnings_dates: Iterable[date] = (),
) -> list[dict[str, Any]]:
    """Compute features using rolling windows ending on each decision date."""

    required = {"Close", "Volume"}
    if not required.issubset(history.columns):
        raise ValueError(f"history requires columns {sorted(required)}")
    frame = history.loc[:, ["Close", "Volume"]].copy().sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if len(frame) < 21:
        return []

    frame["trailing_return_5d"] = frame["Close"].pct_change(5)
    daily_return = frame["Close"].pct_change()
    frame["annualized_volatility_20d"] = daily_return.rolling(20).std() * math.sqrt(252)
    volume_mean = frame["Volume"].rolling(20).mean()
    volume_std = frame["Volume"].rolling(20).std().replace(0, float("nan"))
    frame["volume_zscore_20d"] = (frame["Volume"] - volume_mean) / volume_std

    known_earnings = set(earnings_dates)
    snapshot_id = _snapshot_id(ticker, frame)
    rows = []
    for index, row in frame.dropna().iterrows():
        decision_date = pd.Timestamp(index).date()
        earnings_window = any(
            abs((decision_date - earnings_date).days) <= 1
            for earnings_date in known_earnings
        )
        rows.append(
            {
                "ticker": ticker.upper(),
                "trade_date": decision_date.isoformat(),
                "sector": sector,
                "data_snapshot_id": snapshot_id,
                "information_cutoff": f"{decision_date.isoformat()}T23:59:59Z",
                "trailing_return_5d": float(row["trailing_return_5d"]),
                "annualized_volatility_20d": float(
                    row["annualized_volatility_20d"]
                ),
                "volume_zscore_20d": float(row["volume_zscore_20d"]),
                "earnings_window": earnings_window,
            }
        )
    return rows


def fetch_feature_rows(
    tickers: Sequence[str],
    *,
    start: str,
    end: str,
    sector_by_ticker: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    sectors = {key.upper(): value for key, value in (sector_by_ticker or {}).items()}
    rows = []
    for raw_ticker in tickers:
        ticker = raw_ticker.strip().upper()
        if not ticker:
            continue
        stock = yf.Ticker(ticker)
        history = stock.history(start=start, end=end, auto_adjust=False)
        earnings_dates = _earnings_dates(stock, start=start, end=end)
        rows.extend(
            compute_feature_rows(
                ticker,
                history,
                sector=sectors.get(ticker, "unknown"),
                earnings_dates=earnings_dates,
            )
        )
    return rows


def _earnings_dates(stock: Any, *, start: str, end: str) -> set[date]:
    try:
        values = stock.get_earnings_dates(limit=100)
    except Exception:
        return set()
    if values is None or values.empty:
        return set()
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    return {
        pd.Timestamp(index).date()
        for index in values.index
        if start_date <= pd.Timestamp(index).date() < end_date
    }


def write_feature_rows(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _load_tickers(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True, help="One ticker per line")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sectors", help="Optional JSON object mapping ticker to sector")
    args = parser.parse_args()

    sectors = None
    if args.sectors:
        sectors = json.loads(Path(args.sectors).read_text(encoding="utf-8"))
    rows = fetch_feature_rows(
        _load_tickers(args.tickers),
        start=args.start,
        end=args.end,
        sector_by_ticker=sectors,
    )
    write_feature_rows(rows, args.output)
    print(json.dumps({"feature_rows": len(rows), "output": args.output}))


if __name__ == "__main__":
    main()
