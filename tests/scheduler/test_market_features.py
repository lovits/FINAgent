from datetime import date

import pandas as pd

from training.scheduler.market_features import compute_feature_rows, write_feature_rows


def _history(days: int = 30) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "Close": [100 + index for index in range(days)],
            "Volume": [1000 + index * index for index in range(days)],
        },
        index=index,
    )


def test_features_use_rolling_history_and_mark_earnings_window() -> None:
    rows = compute_feature_rows(
        "aapl",
        _history(),
        sector="Technology",
        earnings_dates={date(2026, 1, 25)},
    )

    assert rows
    row = next(value for value in rows if value["trade_date"] == "2026-01-25")
    assert row["ticker"] == "AAPL"
    assert row["earnings_window"] is True
    assert row["trailing_return_5d"] > 0
    assert row["data_snapshot_id"].startswith("yfinance:AAPL:")


def test_short_history_yields_no_feature_rows() -> None:
    assert compute_feature_rows("AAPL", _history(20)) == []


def test_feature_writer_creates_jsonl(tmp_path) -> None:
    rows = compute_feature_rows("AAPL", _history())
    target = tmp_path / "features.jsonl"
    write_feature_rows(rows, target)
    assert len(target.read_text().splitlines()) == len(rows)
