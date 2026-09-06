"""Date-bounded mainland-China financial news via AKShare/Eastmoney."""

from __future__ import annotations

from datetime import timedelta

import akshare as ak
import pandas as pd

from .config import get_config
from .errors import NoMarketDataError
from .symbol_utils import is_a_share_symbol, normalize_symbol


def _a_share_code(symbol: str) -> str:
    canonical = normalize_symbol(symbol)
    if not is_a_share_symbol(canonical):
        raise NoMarketDataError(symbol, canonical, "AKShare news expects an A-share symbol")
    return canonical.split(".", 1)[0]


def get_news(symbol: str, start_date: str, end_date: str) -> str:
    code = _a_share_code(symbol)
    try:
        with pd.option_context("future.infer_string", False):
            data = ak.stock_news_em(symbol=code)
    except Exception as exc:  # noqa: BLE001 - optional enrichment must degrade
        return f"DATA_UNAVAILABLE: AKShare company news failed ({type(exc).__name__})."
    if data.empty:
        return f"No mainland-China company news found for {normalize_symbol(symbol)}."
    published = pd.to_datetime(data["发布时间"], errors="coerce")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    data = data[(published >= start) & (published < end)].copy()
    data["发布时间"] = published[(published >= start) & (published < end)]
    limit = int(get_config().get("news_article_limit", 20))
    return _render_news(data.sort_values("发布时间", ascending=False).head(limit), normalize_symbol(symbol))


def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 10) -> str:
    try:
        with pd.option_context("future.infer_string", False):
            data = ak.stock_info_global_em()
    except Exception as exc:  # noqa: BLE001 - optional enrichment must degrade
        return f"DATA_UNAVAILABLE: AKShare mainland market news failed ({type(exc).__name__})."
    if data.empty:
        return "No mainland-China market news was returned by AKShare."
    published = pd.to_datetime(data["发布时间"], errors="coerce")
    end = pd.Timestamp(curr_date) + pd.Timedelta(days=1)
    start = end - timedelta(days=look_back_days + 1)
    data = data[(published >= start) & (published < end)].copy()
    data["发布时间"] = published[(published >= start) & (published < end)]
    return _render_global(data.sort_values("发布时间", ascending=False).head(limit))


def get_insider_transactions(symbol: str) -> str:
    _a_share_code(symbol)
    return (
        "DATA_UNAVAILABLE: the free AKShare management-holdings endpoint requires "
        "a full-market crawl and is intentionally disabled for interactive runs."
    )


def _render_news(data: pd.DataFrame, symbol: str) -> str:
    if data.empty:
        return f"No date-valid mainland-China company news found for {symbol}."
    blocks = [f"# Mainland-China company news for {symbol}", "Source: Eastmoney via AKShare"]
    for _, row in data.iterrows():
        content = str(row.get("新闻内容") or "").strip()[:800]
        blocks.append(
            f"## {row.get('新闻标题')}\n"
            f"Published: {row.get('发布时间')} | Publisher: {row.get('文章来源')}\n"
            f"{content}\nLink: {row.get('新闻链接')}"
        )
    return "\n\n".join(blocks)


def _render_global(data: pd.DataFrame) -> str:
    if data.empty:
        return "No date-valid mainland-China market news found."
    blocks = ["# Mainland-China market news", "Source: Eastmoney via AKShare"]
    for _, row in data.iterrows():
        blocks.append(
            f"## {row.get('标题')}\n"
            f"Published: {row.get('发布时间')}\n"
            f"{str(row.get('摘要') or '').strip()[:800]}\nLink: {row.get('链接')}"
        )
    return "\n\n".join(blocks)
