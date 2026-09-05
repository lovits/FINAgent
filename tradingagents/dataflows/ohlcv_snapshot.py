"""Optional immutable OHLCV input for reproducible historical evaluations."""
import json
import os
from pathlib import Path

import pandas as pd

from .symbol_utils import NoMarketDataError, normalize_symbol


def read_ohlcv_snapshot(symbol: str, end_date: str, start_date: str | None = None):
    directory = os.environ.get("TRADINGAGENTS_OHLCV_SNAPSHOT_DIR")
    if not directory:
        return None
    root = Path(directory)
    canonical = normalize_symbol(symbol)
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["symbols"].get(canonical)
    if entry is None:
        raise NoMarketDataError(symbol, canonical, "symbol absent from evaluation snapshot")
    end = pd.Timestamp(end_date)
    if end > pd.Timestamp(entry["cutoff"]):
        raise NoMarketDataError(symbol, canonical, "requested date exceeds frozen task cutoff")
    path = (root / entry["file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("snapshot file must stay inside snapshot directory")
    frame = pd.read_csv(path, parse_dates=["Date"])
    frame = frame[frame["Date"] <= end]
    if start_date is not None:
        frame = frame[frame["Date"] >= pd.Timestamp(start_date)]
    if frame.empty:
        raise NoMarketDataError(symbol, canonical, "no snapshot rows in requested range")
    from .stockstats_utils import _assert_ohlcv_not_stale
    _assert_ohlcv_not_stale(frame, end_date, symbol, canonical)
    return frame.reset_index(drop=True)
