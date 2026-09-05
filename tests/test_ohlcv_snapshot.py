import json

import pandas as pd
import pytest

from tradingagents.dataflows.ohlcv_snapshot import read_ohlcv_snapshot
from tradingagents.dataflows.symbol_utils import NoMarketDataError


@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_OHLCV_SNAPSHOT_DIR", str(tmp_path))
    (tmp_path / "manifest.json").write_text(json.dumps({"symbols": {
        "AMZN": {"file": "AMZN.csv", "cutoff": "2026-07-31"}}}))
    pd.DataFrame({"Date": ["2026-07-29", "2026-07-30", "2026-07-31"],
                  "Open": [1, 2, 3], "High": [2, 3, 4], "Low": [1, 2, 3],
                  "Close": [2, 3, 4], "Volume": [100, 100, 100]}).to_csv(
                      tmp_path / "AMZN.csv", index=False)


def test_snapshot_is_opt_in(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_OHLCV_SNAPSHOT_DIR", raising=False)
    assert read_ohlcv_snapshot("AMZN", "2026-07-31") is None


def test_snapshot_date_filter_and_cutoff(snapshot):
    frame = read_ohlcv_snapshot("AMZN", "2026-07-30", "2026-07-30")
    assert len(frame) == 1 and frame.iloc[0]["Close"] == 3
    with pytest.raises(NoMarketDataError, match="cutoff"):
        read_ohlcv_snapshot("AMZN", "2026-08-01")
    with pytest.raises(NoMarketDataError, match="absent"):
        read_ohlcv_snapshot("UNKNOWN", "2026-07-30")


def test_both_price_and_indicator_inputs_skip_yahoo(snapshot, monkeypatch):
    from tradingagents.dataflows.y_finance import get_YFin_data_online
    from tradingagents.dataflows.stockstats_utils import load_ohlcv

    def forbidden(*args, **kwargs):
        pytest.fail("frozen inputs must not make Yahoo requests")

    monkeypatch.setattr("yfinance.Ticker", forbidden)
    monkeypatch.setattr("yfinance.download", forbidden)
    assert "Frozen historical" in get_YFin_data_online("AMZN", "2026-07-29", "2026-07-30")
    assert len(load_ohlcv("AMZN", "2026-07-30")) == 2


def test_snapshot_rejects_stale_data(snapshot):
    from pathlib import Path
    import os
    manifest = Path(os.environ["TRADINGAGENTS_OHLCV_SNAPSHOT_DIR"]) / "manifest.json"
    manifest.write_text(json.dumps({"symbols": {"AMZN": {
        "file": "AMZN.csv", "cutoff": "2026-09-01"}}}))
    with pytest.raises(NoMarketDataError, match="stale"):
        read_ohlcv_snapshot("AMZN", "2026-09-01")
