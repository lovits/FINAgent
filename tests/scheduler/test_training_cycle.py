import json
from collections import Counter

import pytest

from training.scheduler.sft_dataset import HierarchicalSourceSampler
from training.scheduler.train_with_evaluation import validate_eval_tasks


def test_coverage_sampler_visits_every_sample_once_with_natural_source_ratio():
    rows = [
        {
            "source": "static" if i < 7 else "teacher_audited",
            "task_id": str(i),
            "trajectory_id": str(i),
        }
        for i in range(18)
    ]
    sampler = HierarchicalSourceSampler(rows, cover_all=True)
    indices = list(sampler)
    assert set(indices) == set(range(18))
    assert len(indices) == len(sampler) == 18
    assert Counter(rows[i]["source"] for i in indices) == {"static": 7, "teacher_audited": 11}
    assert list(sampler) == indices
    sampler.set_epoch(1)
    assert list(sampler) != indices


def test_scenario_tasks_are_unique_and_isolated(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"task_id": "train"}) + "\n")
    tasks = [{"task_id": str(i)} for i in range(5)]
    validate_eval_tasks(tasks, path, path)
    tasks[0]["task_id"] = "train"
    with pytest.raises(ValueError, match="overlap"):
        validate_eval_tasks(tasks, path, path)


@pytest.mark.parametrize("phase", ["half", "full"])
@pytest.mark.parametrize("mode_set", ["paired", "three", "learned"])
def test_five_scenarios_review_both_modes_and_preserve_failed_attempts(monkeypatch, tmp_path, phase, mode_set):
    from types import SimpleNamespace

    import torch

    from tests.scheduler.test_extend_sft_sources import _trajectory
    from training.scheduler.train_with_evaluation import evaluate_epoch

    calls = []

    def run_task(task, mode, *args):
        calls.append((task["task_id"], mode))
        trajectory = _trajectory(task["task_id"], mode)
        if task["task_id"] == "0" and mode == "learned":
            trajectory.execution_status = "failed"
        return trajectory

    class Reviewer:
        def assess(self, trajectory):
            return {"status": "reviewed", "quality": 0.75}

    monkeypatch.setattr("training.scheduler.train_with_evaluation.run_task", run_task)
    monkeypatch.setattr("training.scheduler.train_with_evaluation.AutomaticReviewer", Reviewer)
    monkeypatch.setattr(
        "training.scheduler.train_with_evaluation.HFSchedulerPolicy",
        lambda *a, **kw: SimpleNamespace(policy_id="checkpoint"),
    )
    model = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
    evaluate_epoch(
        1,
        model,
        None,
        tmp_path / "checkpoint",
        tasks=[{"task_id": str(i)} for i in range(5)],
        output=tmp_path,
        max_length=32768,
        phase=phase,
        include_teacher=mode_set == "three",
        include_static=mode_set != "learned",
    )
    directory = "epoch-01-half" if phase == "half" else "epoch-01"
    report = json.loads((tmp_path / directory / "comparison.json").read_text())
    assert len(calls) == len(set(calls)) == {"paired": 10, "three": 15, "learned": 5}[mode_set]
    assert report["modes"]["learned"]["completion_rate"] == 0.8
    if mode_set != "learned":
        assert report["quality_summary"]["static"]["reviewed"] == 5
    else:
        assert {mode for _, mode in calls} == {"learned"}
        assert report["baseline_comparison"] is None
        assert "trader_match_rate" not in report["modes"]["learned"]
    assert report["quality_summary"]["learned"]["reviewed"] == 5
    if mode_set == "three":
        assert report["quality_summary"]["teacher"]["reviewed"] == 5
