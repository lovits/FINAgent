import json
from collections import Counter

import pytest
import torch

from tests.scheduler_helpers import FakeTokenizer
from tradingagents.scheduler.actions import SchedulerAction
from training.scheduler.model import register_action_tokens, scheduler_input_ids
from training.scheduler.sft_dataset import (
    HierarchicalSourceSampler,
    MaskedActionCollator,
    SchedulerSFTDataset,
)
from training.scheduler.sft_loss import masked_action_cross_entropy


def _row(index: int, source: str = "static", task_id: str = "task-1") -> dict:
    return {
        "schema_version": "scheduler-sft-v1",
        "sample_id": f"sample-{index}",
        "task_id": task_id,
        "trajectory_id": f"trajectory-{task_id}",
        "step_id": index,
        "source": source,
        "input_text": f"prompt-{index}",
        "valid_actions": ["<ACT_MARKET>", "<ACT_NEWS>"],
        "target_action": "<ACT_NEWS>",
        "input_token_count": 8,
    }


def test_dataset_validates_schema_and_target(tmp_path) -> None:
    target = tmp_path / "sft.jsonl"
    target.write_text(json.dumps(_row(0)) + "\n", encoding="utf-8")
    assert len(SchedulerSFTDataset(target)) == 1

    target.write_text(json.dumps({**_row(0), "target_action": "<ACT_STOP>"}) + "\n")
    with pytest.raises(ValueError, match="target_action"):
        SchedulerSFTDataset(target)


def test_hierarchical_sampler_does_not_let_long_trajectory_dominate() -> None:
    rows = [_row(index, task_id="long") for index in range(100)]
    rows.append(_row(100, task_id="short"))
    sampler = HierarchicalSourceSampler(rows, teacher_probability=0.0)
    selected_tasks = Counter(rows[index]["task_id"] for index in sampler)
    assert selected_tasks["short"] > 25


def test_collator_masks_actions_and_rejects_overflow() -> None:
    tokenizer = FakeTokenizer()
    register_action_tokens(tokenizer)
    collator = MaskedActionCollator(tokenizer, max_length=32)
    batch = collator([_row(0)])
    assert batch["input_ids"].shape[0] == 1
    assert batch["valid_action_mask"].sum().item() == 2

    with pytest.raises(ValueError, match="maximum"):
        collator([{**_row(0), "input_text": "x" * 40}])


def test_masked_cross_entropy_ignores_invalid_high_logit() -> None:
    tokenizer = FakeTokenizer()
    register_action_tokens(tokenizer)
    market = tokenizer.action_ids[SchedulerAction.MARKET]
    news = tokenizer.action_ids[SchedulerAction.NEWS]
    stop = tokenizer.action_ids[SchedulerAction.STOP]
    logits = torch.zeros((1, 2, len(tokenizer)))
    logits[0, 1, market] = 1.0
    logits[0, 1, news] = 3.0
    logits[0, 1, stop] = 100.0
    loss, logprobs = masked_action_cross_entropy(
        logits,
        torch.tensor([1]),
        torch.tensor([[market, news]]),
        torch.tensor([[True, True]]),
        torch.tensor([news]),
    )
    assert loss.item() < 0.2
    assert torch.argmax(logprobs, dim=1).item() == 1


def test_chat_template_batch_encoding_is_normalized_to_token_list() -> None:
    class BatchTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [3, 4, 5], "attention_mask": [1, 1, 1]}

    assert scheduler_input_ids(BatchTokenizer(), "prompt") == [3, 4, 5]
