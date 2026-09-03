"""Create date-filtered, read-only TradingMemoryLog context for paired runs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from tradingagents.agents.utils.memory import TradingMemoryLog


@dataclass(frozen=True)
class MemorySnapshot:
    snapshot_id: str
    context: str
    entry_count: int


def build_memory_snapshot(
    memory_log_path: str | Path,
    *,
    ticker: str,
    trade_date: str,
    same_ticker_limit: int = 5,
    cross_ticker_limit: int = 3,
) -> MemorySnapshot:
    cutoff = date.fromisoformat(trade_date)
    log = TradingMemoryLog({"memory_log_path": str(memory_log_path)})
    eligible = [
        entry
        for entry in log.load_entries()
        if not entry.get("pending")
        and entry.get("date")
        and date.fromisoformat(entry["date"]) < cutoff
    ]
    same = [entry for entry in reversed(eligible) if entry["ticker"] == ticker][
        :same_ticker_limit
    ]
    cross = [entry for entry in reversed(eligible) if entry["ticker"] != ticker][
        :cross_ticker_limit
    ]
    sections = []
    if same:
        sections.append(f"Past analyses of {ticker} (most recent first):")
        sections.extend(_format_entry(entry, include_decision=True) for entry in same)
    if cross:
        sections.append("Recent cross-ticker lessons:")
        sections.extend(_format_entry(entry, include_decision=False) for entry in cross)
    context = "\n\n".join(sections)
    digest = hashlib.sha256(context.encode()).hexdigest()[:16]
    return MemorySnapshot(f"memory:{ticker}:{trade_date}:{digest}", context, len(same) + len(cross))


def _format_entry(entry: dict, *, include_decision: bool) -> str:
    header = f"[{entry['date']} | {entry['ticker']} | {entry['rating']}]"
    parts = [header]
    if include_decision and entry.get("decision"):
        parts.append("DECISION:\n" + entry["decision"])
    if entry.get("reflection"):
        parts.append("REFLECTION:\n" + entry["reflection"])
    return "\n".join(parts)
