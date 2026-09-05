import json
from pathlib import Path
import pytest

from training.scheduler.rl_three_rounds import _task_path, run, validate_plan


def test_plan_requires_eight_distinct_training_tasks(tmp_path):
    training = tmp_path / "train.jsonl"
    common = {"ticker": "AAPL", "trade_date": "2026-01-05", "research_depth": "shallow",
              "output_language": "Chinese", "selected_analysts": ["market"]}
    training.write_text("".join(json.dumps({**common, "task_id": str(i), "split": "train"}) + "\n" for i in range(8)))
    plan = {"train_tasks": str(training)}
    assert len(validate_plan(plan)) == 8
    training.write_text("".join(json.dumps({**common, "task_id": "0", "split": "train"}) + "\n" for _ in range(8)))
    with pytest.raises(ValueError, match="distinct"):
        validate_plan(plan)


def test_round_specific_seed_files_are_disjoint(monkeypatch):
    values = {
        "original": [{"task_id": f"old-{i}"} for i in range(8)],
        "round-4": [{"task_id": f"r4-{i}"} for i in range(8)],
        "round-5": [{"task_id": f"r5-{i}"} for i in range(8)],
    }
    monkeypatch.setattr("training.scheduler.rl_three_rounds.load_tasks",
                        lambda path, split: values[path])
    plan = {"rounds": 5, "train_tasks": "original",
            "train_tasks_by_round": {"4": "round-4", "5": "round-5"}}
    validate_plan(plan)
    assert _task_path(plan, 3) == "original"
    assert _task_path(plan, 4) == "round-4"
    values["round-5"][0] = {"task_id": "r4-0"}
    with pytest.raises(ValueError, match="overlap"):
        validate_plan(plan)


@pytest.mark.parametrize("usable", [True, False])
def test_configured_rounds_use_previous_model_and_skip_updates_without_reward_groups(monkeypatch, tmp_path, usable):
    monkeypatch.setenv("TRADINGAGENTS_OHLCV_SNAPSHOT_DIR", str(tmp_path))
    plan = {"output_dir": str(tmp_path / "run"), "snapshot_dir": str(tmp_path),
            "initial_adapter": "sft-six", "train_tasks": "eight", "base_model": "tiny",
            "rounds": 5}
    monkeypatch.setattr("training.scheduler.rl_three_rounds.validate_plan", lambda p: [])
    events = []

    def stage(module, config, directory):
        name = module.rsplit(".", 1)[-1]
        events.append(name)
        if name == "collect_rollouts":
            round_number = len([e for e in events if e == name])
            expected = "sft-six" if round_number == 1 else str(Path(plan["output_dir"]) / f"round-{round_number-1}/checkpoint")
            assert config.active_adapter_path == expected
            assert config.reference_adapter_path == "sft-six"
            assert config.reward_mode == "automatic" and config.group_size == 4
            target = directory / "rollouts"
            target.mkdir()
            (target / "rollout_manifest.json").write_text(json.dumps({"counts": {
                "usable_groups": 8 if usable else 0,
                "trajectories": 32 if usable else 0, "rows": 100 if usable else 0}}))
        else:
            assert config.epochs == 1

    monkeypatch.setattr("training.scheduler.rl_three_rounds.run_stage", stage)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    if usable:
        run(plan)
        assert events == ["collect_rollouts", "train_grpo"] * 5
    else:
        with pytest.raises(RuntimeError, match="no parameter update"):
            run(plan)
        assert events == ["collect_rollouts"]
