from pathlib import Path

from training.scheduler.validate_task_manifest import validate_task_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_recent_static_langgraph_pilot_manifest_is_valid():
    report = validate_task_manifest(
        ROOT / "data/scheduler/tasks/static_langgraph_pilot_v1.jsonl",
        ROOT / "data/scheduler/tasks/static_langgraph_pilot_v1.manifest.json",
    )

    assert report["tasks"] == 18
    assert report["tickers"] == 11
    assert report["market_window"] == ["2026-04-01", "2026-08-28"]
    assert set(report["seed_families"]) == {
        "earnings_window",
        "positive_momentum",
        "negative_momentum",
        "high_volatility",
        "volume_shock",
        "quiet_control",
    }
