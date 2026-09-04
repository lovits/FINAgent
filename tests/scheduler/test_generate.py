import json

import pytest

from tradingagents.scheduler.trajectory import SchedulerStep, SchedulerTrajectory
from training.scheduler.environment import EnvironmentRunResult
from training.scheduler.generate import generate_trajectories, load_tasks


def _complete_trajectory(identifier: str, mode: str, run_id: str) -> SchedulerTrajectory:
    trajectory = SchedulerTrajectory(
        identifier,
        run_id,
        "task-1",
        mode,  # type: ignore[arg-type]
        "static-v1" if mode == "static" else "teacher-v1",
        "AAPL",
        "2026-01-05",
        data_snapshot_id="snapshot-1",
    )
    trajectory.add_step(
        SchedulerStep(
            0,
            {"news_report": ""},
            f"{mode}-prompt",
            ["<ACT_NEWS>"],
            "<ACT_NEWS>",
            "News Analyst",
            state_after={"news_report": "evidence"},
        )
    )
    final_state = {
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
        "investment_plan": "Hold",
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
        "final_trade_decision": "**Rating**: Hold",
    }
    trajectory.add_step(
        SchedulerStep(
            1,
            final_state,
            f"{mode}-stop-prompt",
            ["<ACT_STOP>"],
            "<ACT_STOP>",
            None,
            state_after=final_state,
        )
    )
    trajectory.final_outputs = {
        "investment_plan": final_state["investment_plan"],
        "trader_investment_plan": final_state["trader_investment_plan"],
        "final_trade_decision": final_state["final_trade_decision"],
    }
    trajectory.execution_status = "completed"
    return trajectory


class _Environment:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, task, *, mode, run_id, trajectory_id, **kwargs):
        trajectory = _complete_trajectory(trajectory_id, mode, run_id)
        return EnvironmentRunResult(trajectory, trajectory.steps[-1].state_before)


def _task() -> dict:
    return {
        "task_id": "task-1",
        "ticker": "AAPL",
        "trade_date": "2026-01-05",
        "asset_type": "stock",
        "split": "train",
        "dataset_version": "scheduler-tasks-v2",
        "data_snapshot_id": "snapshot-1",
        "selected_analysts": ["market"],
        "research_depth": "shallow",
        "output_language": "Chinese",
    }


def test_generation_writes_audited_trajectory_and_manifest(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("training.scheduler.generate.TradingAgentsSchedulerEnvironment", _Environment)
    counts = generate_trajectories(
        [_task()],
        output_dir=tmp_path,
        mode="static",
        run_id="run-1",
        config={},
    )

    assert counts == {
        "accepted": 1,
        "rejected": 0,
        "skipped": 0,
        "completed": 1,
        "failed": 0,
        "fallback": 0,
        "budget_exhausted": 0,
        "context_overflow": 0,
    }
    assert len((tmp_path / "raw.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "accepted.jsonl").read_text().splitlines()) == 1
    manifest = json.loads((tmp_path / "generation_manifest.json").read_text())
    assert manifest["trajectories_per_task"] == 1
    assert manifest["task_count"] == 1
    assert manifest["schema_versions"]["actions"] == "scheduler-actions-v1"
    assert manifest["orchestration_profile_version"] == (
        "multi-analyst-shallow-v1"
    )
    assert manifest["task_inputs"] == {
        "analyst_sets": [["market"]],
        "output_languages": ["Chinese"],
        "research_depths": ["shallow"],
    }
    assert len(manifest["generation_config_hashes"]) == 1
    assert len(manifest["generation_config_hashes"][0]) == 64
    assert "git_commit" in manifest
    assert manifest["dataset_counts"]["total"] == 1
    assert manifest["dataset_counts"]["completed"] == 1

    resumed = generate_trajectories(
        [_task()],
        output_dir=tmp_path,
        mode="static",
        run_id="run-1",
        config={},
        resume=True,
    )
    assert resumed["skipped"] == 1
    resumed_manifest = json.loads(
        (tmp_path / "generation_manifest.json").read_text()
    )
    assert resumed_manifest["counts"]["skipped"] == 1
    assert resumed_manifest["dataset_counts"]["completed"] == 1


def test_generation_requires_resume_for_existing_record(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("training.scheduler.generate.TradingAgentsSchedulerEnvironment", _Environment)
    arguments = {
        "tasks": [_task()],
        "output_dir": tmp_path,
        "mode": "static",
        "run_id": "run-1",
        "config": {},
    }
    generate_trajectories(**arguments)
    with pytest.raises(FileExistsError, match="--resume"):
        generate_trajectories(**arguments)


def test_load_tasks_filters_split(tmp_path) -> None:
    target = tmp_path / "tasks.jsonl"
    target.write_text(
        json.dumps(_task()) + "\n" + json.dumps({**_task(), "task_id": "task-2", "split": "validation"}) + "\n"
    )
    assert [task["task_id"] for task in load_tasks(target, split="train")] == ["task-1"]
