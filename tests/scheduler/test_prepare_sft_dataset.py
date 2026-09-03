import copy
import json

import pytest

from tradingagents.scheduler.actions import SchedulerAction
from training.scheduler.dataset import SchedulerSFTDataset
from training.scheduler.prepare_sft_dataset import prepare_sft_dataset


def _state(ticker, trade_date, actions):
    return json.dumps(
        {
            "task": {"ticker": ticker, "trade_date": trade_date},
            "valid_actions": actions,
        }
    )


def _record(task_id, ticker, task_split):
    trade_date = "2026-08-26"
    return {
        "record_type": "static_langgraph",
        "dataset_version": "v1",
        "action_schema_version": "v1",
        "state_schema_version": "v1",
        "generation_key": f"{task_id}:static:0:v1",
        "trajectory_id": f"trajectory-{task_id}",
        "sample_index": 0,
        "status": "accepted",
        "task": {
            "task_id": task_id,
            "ticker": ticker,
            "trade_date": trade_date,
            "split": task_split,
            "seed_family": "earnings_window",
            "sector": "information_technology",
        },
        "provenance": {
            "graph_mode": "static",
            "information_cutoff": trade_date,
            "code_commit": "abc123",
        },
        "node_steps": [{"step_id": 0}, {"step_id": 1}],
        "scheduler_examples": [
            {
                "step_id": 0,
                "input_text": _state(
                    ticker, trade_date, [SchedulerAction.MARKET.value]
                ),
                "valid_actions": [SchedulerAction.MARKET.value],
                "target_action": SchedulerAction.MARKET.value,
                "source_node": "Market Analyst",
                "action_valid": True,
            },
            {
                "step_id": 1,
                "input_text": _state(
                    ticker, trade_date, [SchedulerAction.STOP.value]
                ),
                "valid_actions": [SchedulerAction.STOP.value],
                "target_action": SchedulerAction.STOP.value,
                "source_node": "END",
                "action_valid": True,
            },
        ],
        "final_outputs": {"trader_action": "Buy", "portfolio_rating": "Buy"},
        "labels": {"completed": True, "data_sparse": False},
        "quality_control": {
            "static_actions_valid": True,
            "failure_reason": None,
            "human_review": "approved",
        },
    }


def test_prepare_sft_dataset_audits_splits_and_writes_manifest(tmp_path):
    source = tmp_path / "accepted.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                _record("train-task", "NVDA", "train"),
                _record("test-task", "ORCL", "test"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "sft"
    manifest = prepare_sft_dataset(
        [source],
        output,
        dataset_id="scheduler-sft-fixture",
        created_at="2026-09-03T00:00:00+00:00",
    )

    assert manifest["eligible_trajectories"] == 2
    assert manifest["splits"]["train"]["examples"] == 2
    assert manifest["splits"]["test"]["examples"] == 2
    assert manifest["observation_loss_masked"] is True
    assert len(SchedulerSFTDataset(output / "train.jsonl")) == 2
    first = json.loads((output / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["schema_version"] == "v1"
    assert first["metadata"]["task_split"] == "train"
    assert first["metadata"]["trajectory_id"] == "trajectory-train-task"


def test_prepare_sft_dataset_blocks_invalid_accepted_trajectory(tmp_path):
    record = copy.deepcopy(_record("bad", "NVDA", "train"))
    record["scheduler_examples"].pop()
    source = tmp_path / "accepted.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = tmp_path / "sft"

    with pytest.raises(ValueError, match="failed audit"):
        prepare_sft_dataset([source], output, dataset_id="bad")

    report = json.loads((output / "audit_report.json").read_text(encoding="utf-8"))
    assert report["blocking_records"] == 1
    assert not (output / "manifest.json").exists()


def test_prepare_sft_dataset_blocks_ticker_split_leakage(tmp_path):
    source = tmp_path / "accepted.jsonl"
    source.write_text(
        json.dumps(_record("train", "NVDA", "train"))
        + "\n"
        + json.dumps(_record("test", "NVDA", "test"))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ticker-disjoint"):
        prepare_sft_dataset([source], tmp_path / "sft", dataset_id="leak")
