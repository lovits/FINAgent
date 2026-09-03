from tradingagents.scheduler.store import TrajectoryStore
from tradingagents.scheduler.trajectory import SchedulerStep, SchedulerTrajectory
from training.scheduler.environment import EnvironmentRunResult
from training.scheduler.generate import generate_trajectories
from training.scheduler.pair_dataset import pair_datasets, write_comparisons
from training.scheduler.prepare_sft import prepare_sft_dataset


def _task() -> dict:
    return {
        "task_id": "task-1",
        "ticker": "AAPL",
        "trade_date": "2026-01-05",
        "asset_type": "stock",
        "split": "train",
        "data_snapshot_id": "snapshot-1",
    }


class _Environment:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, task, *, mode, run_id, trajectory_id, **kwargs):
        trajectory = SchedulerTrajectory(
            trajectory_id,
            run_id,
            task["task_id"],
            mode,
            f"{mode}-v1",
            task["ticker"],
            task["trade_date"],
            data_snapshot_id=task["data_snapshot_id"],
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
        return EnvironmentRunResult(trajectory, final_state)


def test_mock_data_pipeline_reaches_sft_jsonl(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("training.scheduler.generate.TradingAgentsSchedulerEnvironment", _Environment)
    static_dir = tmp_path / "static"
    teacher_dir = tmp_path / "teacher"
    common = {
        "tasks": [_task()],
        "run_id": "run-1",
        "config": {},
        "selected_analysts": ("market", "social", "news", "fundamentals"),
    }
    generate_trajectories(output_dir=static_dir, mode="static", **common)
    generate_trajectories(output_dir=teacher_dir, mode="teacher", **common)

    static = TrajectoryStore(static_dir / "accepted.jsonl").load()
    teacher = TrajectoryStore(teacher_dir / "accepted.jsonl").load()
    comparisons = pair_datasets(static, teacher)
    write_comparisons(comparisons, tmp_path / "paired" / "comparison.jsonl")
    examples, overflow = prepare_sft_dataset(
        static,
        teacher,
        verified_ids={comparisons[0].teacher_trajectory_id},
        token_counter=len,
    )

    assert comparisons[0].pair_status == "teacher_verified"
    assert {example.source for example in examples} == {"static", "teacher_verified"}
    assert len(examples) == 4
    assert overflow == 0
