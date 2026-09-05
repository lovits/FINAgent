import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training.scheduler.rl_three_rounds import run, validate_plan


def test_plan_requires_eight_train_and_two_disjoint_evaluation_tasks(tmp_path):
    training = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    common = {"ticker": "AAPL", "trade_date": "2026-01-05", "research_depth": "shallow",
              "output_language": "Chinese", "selected_analysts": ["market"]}
    training.write_text("".join(json.dumps({**common, "task_id": str(i), "split": "train"}) + "\n" for i in range(8)))
    evaluation.write_text("".join(json.dumps({**common, "task_id": str(i), "split": "validation"}) + "\n" for i in (8, 9)))
    plan = {"train_tasks": str(training), "eval_tasks": str(evaluation)}
    assert len(validate_plan(plan)) == 2
    evaluation.write_text(json.dumps({**common, "task_id": "0"}) + "\n" + json.dumps({**common, "task_id": "9"}) + "\n")
    with pytest.raises(ValueError, match="leakage"):
        validate_plan(plan)


@pytest.mark.parametrize("usable", [True, False])
def test_three_rounds_use_previous_model_and_skip_updates_without_reward_groups(monkeypatch, tmp_path, usable):
    monkeypatch.setenv("TRADINGAGENTS_OHLCV_SNAPSHOT_DIR", str(tmp_path))
    plan = {"output_dir": str(tmp_path / "run"), "snapshot_dir": str(tmp_path),
            "initial_adapter": "sft-six", "train_tasks": "eight", "base_model": "tiny"}
    tasks = [{"task_id": "eval-A"}, {"task_id": "eval-B"}]
    monkeypatch.setattr("training.scheduler.rl_three_rounds.validate_plan", lambda p: tasks)
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
                "trajectories": 32 if usable else 0, "rows": 100 if usable else 0}}))
        else:
            assert config.epochs == 1

    def evaluate(*args, **kwargs):
        assert kwargs["tasks"] is tasks
        assert not kwargs["include_static"] and not kwargs["include_teacher"]
        events.append("evaluate")
        target = kwargs["output"] / f"epoch-{args[0]:02d}"
        target.mkdir(parents=True)
        (target / "comparison.json").write_text(json.dumps({"modes": {"learned": {"completion_rate": 1}}}))

    monkeypatch.setattr("training.scheduler.rl_three_rounds.run_stage", stage)
    monkeypatch.setattr("training.scheduler.rl_three_rounds.evaluate_epoch", evaluate)
    monkeypatch.setattr("training.scheduler.rl_three_rounds.load_scheduler_model", lambda *a, **kw: (
        SimpleNamespace(to=lambda device: None, config=SimpleNamespace()), None, None))
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    if usable:
        run(plan)
        assert events == ["collect_rollouts", "train_grpo", "evaluate"] * 3
    else:
        with pytest.raises(RuntimeError, match="no parameter update"):
            run(plan)
        assert events == ["collect_rollouts"]
