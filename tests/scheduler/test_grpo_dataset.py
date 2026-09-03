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
    target.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    dataset = SchedulerGRPODataset(target)
    tokenizer = FakeTokenizer()
    register_action_tokens(tokenizer)
    batch = GRPOCollator(tokenizer, max_length=64)([dataset[0]])

    assert batch["old_logprobs"].item() == -0.5
    assert batch["ref_logprobs"].item() == pytest.approx(-0.7)
    assert batch["advantages"].item() == 1.0
