import json

from tradingagents.scheduler.policy import CallableSchedulerPolicy
from tradingagents.scheduler.trajectory import SchedulerStep, SchedulerTrajectory
from training.scheduler.collect_evaluation import (
    EvaluationCollectionConfig,
    collect,
)
from training.scheduler.environment import EnvironmentRunResult


class _Environment:
    runtime_config = None
    selected_analysts = None

    def __init__(self, config, **kwargs):
        type(self).runtime_config = config
        type(self).selected_analysts = kwargs["selected_analysts"]

    def run(self, task, *, mode, run_id, policy, trajectory_id, **kwargs):
        trajectory = SchedulerTrajectory(
            trajectory_id,
            run_id,
            task["task_id"],
            "learned",
            policy.policy_id,
            task["ticker"],
            task["trade_date"],
            data_snapshot_id=task["data_snapshot_id"],
        )
        state = {"news_report": ""}
        trajectory.add_step(
            SchedulerStep(
                0,
                state,
                "news prompt",
                ["<ACT_NEWS>"],
                "<ACT_NEWS>",
                "News Analyst",
                state_after={"news_report": "evidence"},
            )
        )
        final_state = {
            "investment_plan": "Hold",
            "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: HOLD",
            "final_trade_decision": "**Rating**: Hold",
        }
        if "market" in task["selected_analysts"]:
            final_state["market_report"] = "market evidence"
        if "news" in task["selected_analysts"]:
            final_state["news_report"] = "news evidence"
        trajectory.add_step(
            SchedulerStep(
                1,
                final_state,
                "stop prompt",
                ["<ACT_STOP>"],
                "<ACT_STOP>",
                None,
                state_after=final_state,
            )
        )
        trajectory.final_outputs = dict(final_state)
        trajectory.provenance = {
            "selected_analysts": task["selected_analysts"],
            "research_depth": task["research_depth"],
            "output_language": task["output_language"],
            "expert_config_hash": "+".join(task["selected_analysts"]),
            "generation_config_hash": "generation-config",
        }
        trajectory.execution_status = "completed"
        return EnvironmentRunResult(trajectory, final_state)


def _task(
    task_id: str, ticker: str, snapshot: str, selected_analysts: list[str]
) -> dict:
    return {
        "task_id": task_id,
        "ticker": ticker,
        "trade_date": "2026-01-05",
        "data_snapshot_id": snapshot,
        "split": "validation",
        "selected_analysts": selected_analysts,
        "research_depth": "shallow",
        "output_language": "Chinese",
    }


def test_collect_evaluation_writes_one_deterministic_trajectory_per_task(
    monkeypatch, tmp_path
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(
            json.dumps(task)
            for task in (
                _task("task-1", "AAPL", "snapshot-1", ["market"]),
                _task("task-2", "MSFT", "snapshot-2", ["market", "news"]),
            )
        )
        + "\n"
    )
    policy = CallableSchedulerPolicy(lambda context: context.valid_actions[0], policy_id="sft")
    loaded_configs = []

    def load_policy(config):
        loaded_configs.append(config)
        return policy

    monkeypatch.setattr(
        "training.scheduler.collect_evaluation.load_hf_scheduler_policy",
        load_policy,
    )
    monkeypatch.setattr(
        "training.scheduler.collect_evaluation.TradingAgentsSchedulerEnvironment",
        _Environment,
    )

    counts = collect(
        EvaluationCollectionConfig(
            tasks_path=str(tasks_path),
            adapter_path="adapter/sft",
            output_dir=str(tmp_path / "evaluation"),
            run_id="sft-validation",
            base_model="tiny",
            base_revision=None,
        )
    )

    assert counts == {
        "accepted": 2,
        "rejected": 0,
        "skipped": 0,
        "completed": 2,
        "failed": 0,
        "fallback": 0,
        "budget_exhausted": 0,
        "context_overflow": 0,
    }
    raw_rows = (tmp_path / "evaluation" / "raw.jsonl").read_text().splitlines()
    assert len(raw_rows) == 2
    manifest = json.loads(
        (tmp_path / "evaluation" / "collection_manifest.json").read_text()
    )
    assert manifest["task_count"] == 2
    assert manifest["policy_id"] == "sft"
    assert manifest["scheduler_temperature"] == 0.0
    assert manifest["expert_temperature"] == 0.0
    assert manifest["dataset_counts"]["total"] == 2
    assert manifest["dataset_counts"]["completed"] == 2
    assert manifest["task_inputs"]["analyst_sets"] == [
        ["market"],
        ["market", "news"],
    ]
    assert loaded_configs[0]["scheduler_adapter_path"] == "adapter/sft"
    assert _Environment.runtime_config["orchestration_mode"] == "learned"
    assert _Environment.runtime_config["output_language"] == "Chinese"
    assert _Environment.selected_analysts == ("market", "news")
