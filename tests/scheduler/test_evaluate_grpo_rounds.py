import json

import pytest

from training.scheduler.evaluate_grpo_rounds import summarize, validate_tasks


def test_requires_two_held_out_tasks():
    training = [{"task_id": "train"}]
    evaluation = [{"task_id": "a"}, {"task_id": "b"}]
    validate_tasks(training, evaluation)
    with pytest.raises(ValueError, match="overlap"):
        validate_tasks(training, [{"task_id": "train"}, {"task_id": "b"}])
    with pytest.raises(ValueError, match="exactly two"):
        validate_tasks(training, [{"task_id": "a"}])


def test_summary_keeps_rounds_separate(monkeypatch):
    from tests.scheduler.test_extend_sft_sources import _trajectory

    first = _trajectory("a", "learned")
    first.trajectory_id = "rl1-a"
    second = _trajectory("a", "learned")
    second.trajectory_id = "rl2-a"
    reviews = {
        "rl1-a": {"status": "reviewed", "quality": 0.5},
        "rl2-a": {"status": "reviewed", "quality": 0.75},
    }
    monkeypatch.setattr(
        "training.scheduler.evaluate_grpo_rounds.automatic_reward",
        lambda trajectory, review: type("Reward", (), {"total": review["quality"]})(),
    )
    result = summarize({"rl-1": [first], "rl-2": [second]}, reviews)
    assert result["rl-1"]["mean_quality"] == 0.5
    assert result["rl-2"]["mean_quality"] == 0.75
    assert result["rl-1"]["task_results"][0]["task_id"] == "a"
