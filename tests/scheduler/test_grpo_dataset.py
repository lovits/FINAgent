import json

import pytest

from tests.scheduler_helpers import FakeTokenizer
from training.scheduler.grpo_dataset import GRPOCollator, SchedulerGRPODataset
from training.scheduler.model import register_action_tokens


def _row() -> dict:
    return {
        "schema_version": "scheduler-grpo-v1",
        "rollout_group_id": "group-1",
        "trajectory_id": "trajectory-1",
        "task_id": "task-1",
        "step_id": 0,
        "serialized_state": "choose",
        "valid_actions": ["<ACT_MARKET>", "<ACT_NEWS>"],
        "selected_action": "<ACT_NEWS>",
        "old_logprob": -0.5,
        "ref_logprob": -0.7,
        "reward_total": 0.4,
        "reward_components": {},
        "advantage": 1.0,
    }


def test_grpo_dataset_and_collator_preserve_policy_statistics(tmp_path) -> None:
    target = tmp_path / "grpo.jsonl"
    second = {**_row(), "trajectory_id": "trajectory-2", "advantage": -1.0}
    target.write_text(json.dumps(_row()) + "\n" + json.dumps(second) + "\n",
                      encoding="utf-8")
    dataset = SchedulerGRPODataset(target, expected_group_size=2)
    tokenizer = FakeTokenizer()
    register_action_tokens(tokenizer)
    batch = GRPOCollator(tokenizer, max_length=64)([dataset[0]])

    assert batch["old_logprobs"].item() == -0.5
    assert batch["ref_logprobs"].item() == pytest.approx(-0.7)
    assert batch["advantages"].item() == 1.0


def test_grpo_dataset_rejects_singleton_group(tmp_path) -> None:
    target = tmp_path / "grpo.jsonl"
    target.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        SchedulerGRPODataset(target)


def test_grpo_dataset_requires_complete_zero_mean_groups(tmp_path) -> None:
    target = tmp_path / "grpo.jsonl"
    first = _row()
    first["advantage"] = 1.0
    second = {
        **_row(),
        "trajectory_id": "trajectory-2",
        "advantage": -1.0,
    }
    target.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    dataset = SchedulerGRPODataset(target, expected_group_size=2)
    assert len(dataset) == 2

    with pytest.raises(ValueError, match="expected 4"):
        SchedulerGRPODataset(target, expected_group_size=4)


def test_grpo_dataset_rejects_non_zero_mean_credit(tmp_path) -> None:
    target = tmp_path / "grpo.jsonl"
    first = _row()
    second = {**_row(), "trajectory_id": "trajectory-2"}
    target.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not zero-mean"):
        SchedulerGRPODataset(target, expected_group_size=2)


def test_each_trajectory_has_equal_total_loss_weight_despite_length(tmp_path):
    import torch

    from training.scheduler.grpo_loss import grpo_clipped_loss

    rows = [_row()] + [{**_row(), "trajectory_id": "long", "step_id": step,
                       "advantage": -1.0} for step in range(3)]
    path = tmp_path / "groups.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = SchedulerGRPODataset(path, expected_group_size=2)
    weights = torch.tensor([row["loss_weight"] for row in dataset])
    assert weights[0] == weights[1:].sum()
    loss = grpo_clipped_loss(torch.zeros(4), torch.zeros(4), torch.zeros(4),
                             torch.tensor([1., -1., -1., -1.]), loss_weights=weights)
    assert float(loss.policy_loss) == pytest.approx(0, abs=1e-6)
