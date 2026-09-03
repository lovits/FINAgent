from training.scheduler.memory_snapshot import build_memory_snapshot


def test_memory_snapshot_excludes_future_and_pending_entries(tmp_path) -> None:
    source = tmp_path / "memory.md"
    source.write_text(
        """[2025-12-01 | AAPL | Hold | +1.0% | +0.5% | 5d]

DECISION:
old decision

REFLECTION:
old lesson

<!-- ENTRY_END -->

[2026-02-01 | AAPL | Buy | +2.0% | +1.0% | 5d]

DECISION:
future decision

<!-- ENTRY_END -->

[2025-12-15 | MSFT | Hold | pending]

DECISION:
pending decision

<!-- ENTRY_END -->
""",
        encoding="utf-8",
    )
    snapshot = build_memory_snapshot(
        source,
        ticker="AAPL",
        trade_date="2026-01-05",
    )

    assert "old decision" in snapshot.context
    assert "old lesson" in snapshot.context
    assert "future decision" not in snapshot.context
    assert "pending decision" not in snapshot.context
    assert snapshot.entry_count == 1
