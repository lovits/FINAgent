import json

import pytest

from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from tradingagents.scheduler.trajectory import SchedulerTrajectory


def _trajectory(identifier: str) -> SchedulerTrajectory:
    return SchedulerTrajectory(
        trajectory_id=identifier,
        run_id="run-1",
        task_id="task-1",
        mode="static",
        policy_id="static-v1",
        ticker="AAPL",
        trade_date="2026-01-05",
    )


def test_store_appends_loads_and_deduplicates(tmp_path) -> None:
    store = TrajectoryStore(tmp_path / "trajectories.jsonl")
    store.append(_trajectory("one"))
    store.append(_trajectory("two"))

    assert [value.trajectory_id for value in store.load()] == ["one", "two"]
    with pytest.raises(ValueError, match="already exists"):
        store.append(_trajectory("one"))


def test_store_rejects_invalid_json(tmp_path) -> None:
    target = tmp_path / "bad.jsonl"
    target.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        list(TrajectoryStore(target).iter_dicts())


def test_atomic_json_write_replaces_complete_document(tmp_path) -> None:
    target = tmp_path / "manifest.json"
    write_json_atomic(target, {"count": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"count": 2}
    assert not (tmp_path / "manifest.json.tmp").exists()
