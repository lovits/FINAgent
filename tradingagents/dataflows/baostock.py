"""Free mainland-China market data backed by BaoStock's own data service."""

from __future__ import annotations

import contextlib
import io
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import baostock as bs
import pandas as pd

from .config import get_config
from .errors import NoMarketDataError
from .symbol_utils import is_a_share_symbol, normalize_symbol
from .utils import safe_ticker_component

_SESSION_LOCK = threading.RLock()
_OHLCV_FIELDS = (
    "date,code,open,high,low,close,volume,amount,turn,tradestatus,"
    "pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)
_CACHE_TTL_SECONDS = 900


def to_baostock_code(symbol: str) -> str:
    canonical = normalize_symbol(symbol)
    if not is_a_share_symbol(canonical):
        raise NoMarketDataError(symbol, canonical, "BaoStock only supports A-share symbols")
    code, suffix = canonical.split(".", 1)
    exchange = {"SS": "sh", "SZ": "sz"}[suffix]
    return f"{exchange}.{code}"


@contextmanager
def _session() -> Iterator[None]:
    with _SESSION_LOCK, contextlib.redirect_stdout(io.StringIO()):
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
        try:
            yield
        finally:
            bs.logout()


def _frame(result) -> pd.DataFrame:
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock query failed: {result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def _history(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    code = to_baostock_code(symbol)
    with _session():
        data = _frame(
            bs.query_history_k_data_plus(
                code,
                _OHLCV_FIELDS,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
        )
    if data.empty:
        raise NoMarketDataError(symbol, normalize_symbol(symbol), "BaoStock returned no rows")
    data = data.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "amount": "Amount",
            "turn": "Turnover",
            "pctChg": "PctChange",
        }
    )
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    numeric = ["Open", "High", "Low", "Close", "Volume", "Amount", "Turnover", "PctChange"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    return data.dropna(subset=["Date", "Close"]).reset_index(drop=True)


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Five-year adjusted A-share OHLCV with the same cache contract as Yahoo."""

    end = pd.Timestamp(curr_date)
    start = end - pd.DateOffset(years=5)
    cache_dir = Path(get_config()["data_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = safe_ticker_component(normalize_symbol(symbol))
    path = cache_dir / f"{safe}-BaoStock-data-{start:%Y-%m-%d}-{end:%Y-%m-%d}.csv"
    historical = end.date() < pd.Timestamp.today().date()
    if path.exists() and (historical or (os.path.getmtime(path) + _CACHE_TTL_SECONDS > datetime.now().timestamp())):
        return pd.read_csv(path, parse_dates=["Date"])
    data = _history(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    data.to_csv(path, index=False, encoding="utf-8")
    return data


def get_stock(symbol: str, start_date: str, end_date: str) -> str:
    data = _history(symbol, start_date, end_date)
    header = (
        f"# A-share data for {normalize_symbol(symbol)} from {start_date} to {end_date}\n"
        f"# Source: BaoStock; adjusted with forward-adjust factor\n"
        f"# Total records: {len(data)}\n\n"
    )
    return header + data.to_csv(index=False)


def get_fundamentals(symbol: str, curr_date: str) -> str:
    code = to_baostock_code(symbol)
    start = (pd.Timestamp(curr_date) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    with _session():
        basic = _frame(bs.query_stock_basic(code=code))
        industry = _frame(bs.query_stock_industry(code=code, date=curr_date))
        quote = _frame(
            bs.query_history_k_data_plus(
                code,
                "date,code,close,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                start_date=start,
                end_date=curr_date,
                frequency="d",
                adjustflag="2",
            )
        )
    sections = ["# A-share Fundamentals", "Source: BaoStock"]
    if not basic.empty:
        sections.extend(("## Company", basic.to_markdown(index=False)))
    if not industry.empty:
        sections.extend(("## Industry", industry.to_markdown(index=False)))
    if not quote.empty:
        sections.extend(("## Latest valuation and trading metrics", quote.tail(1).to_markdown(index=False)))
    return "\n\n".join(sections)


def get_identity(symbol: str) -> dict[str, str]:
    code = to_baostock_code(symbol)
    with _session():
        basic = _frame(bs.query_stock_basic(code=code))
        industry = _frame(bs.query_stock_industry(code=code))
    identity: dict[str, str] = {"exchange": code.split(".", 1)[0].upper()}
    if not basic.empty and str(basic.iloc[0].get("code_name") or "").strip():
        identity["company_name"] = str(basic.iloc[0]["code_name"]).strip()
    if not industry.empty and str(industry.iloc[0].get("industry") or "").strip():
        identity["industry"] = str(industry.iloc[0]["industry"]).strip()
    return identity


def get_balance_sheet(symbol: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _quarterly_report(symbol, curr_date, freq, "Balance Sheet", bs.query_balance_data)


def get_cashflow(symbol: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _quarterly_report(symbol, curr_date, freq, "Cash Flow", bs.query_cash_flow_data)


def get_income_statement(symbol: str, freq: str = "quarterly", curr_date: str | None = None) -> str:
    return _quarterly_report(symbol, curr_date, freq, "Profitability", bs.query_profit_data)


def _quarterly_report(symbol: str, curr_date: str | None, freq: str, title: str, query) -> str:
    cutoff = pd.Timestamp(curr_date or datetime.now().date())
    code = to_baostock_code(symbol)
    quarters = _candidate_quarters(cutoff, annual=freq.lower() == "annual")
    with _session():
        for year, quarter in quarters:
            data = _frame(query(code, year, quarter))
            if data.empty:
                continue
            if "pubDate" in data:
                published = pd.to_datetime(data["pubDate"], errors="coerce")
                data = data[published <= cutoff]
            if not data.empty:
                return f"# A-share {title}\n\nSource: BaoStock\n\n{data.to_markdown(index=False)}"
    return f"DATA_UNAVAILABLE: BaoStock has no date-valid {title.lower()} for {symbol}."


def _candidate_quarters(cutoff: pd.Timestamp, *, annual: bool) -> list[tuple[int, int]]:
    if annual:
        return [(year, 4) for year in range(cutoff.year - 1, cutoff.year - 6, -1)]
    year, quarter = cutoff.year, (cutoff.month - 1) // 3 + 1
    values = []
    for _ in range(8):
        values.append((year, quarter))
        quarter -= 1
        if quarter == 0:
            year, quarter = year - 1, 4
    return values
