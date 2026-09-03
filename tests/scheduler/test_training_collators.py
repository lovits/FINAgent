import torch

from training.scheduler.collator import IGNORE_INDEX, ActionOnlyCollator, encode_sft_example
from training.scheduler.grpo_dataset import GRPOCollator


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        if text == "<ACT_MARKET>":
            return {"input_ids": [9]}
        return {"input_ids": [1, 3, 4]}


def test_sft_encoding_masks_prompt_and_supervises_action_and_eos():
    encoded = encode_sft_example(
        _Tokenizer(), "state", "<ACT_MARKET>", max_length=16
    )

    assert encoded["labels"][:3] == [IGNORE_INDEX] * 3
    assert encoded["labels"][-2:] == [9, 2]
    batch = ActionOnlyCollator(_Tokenizer(), max_length=16)(
        [{"input_text": "state", "target_action": "<ACT_MARKET>"}]
    )
    assert batch["input_ids"].shape == batch["labels"].shape


def test_grpo_collator_has_exactly_one_policy_token_per_row():
    batch = GRPOCollator(_Tokenizer(), max_length=16)(
        [
            {
                "serialized_state": "state",
                "selected_action": "<ACT_MARKET>",
                "valid_actions": ["<ACT_MARKET>"],
                "advantage": 1.0,
                "old_logprob": -0.2,
                "ref_logprob": -0.3,
            }
        ]
    )

    assert batch["action_mask"].shape[1] == batch["input_ids"].shape[1] - 1
    assert int(batch["action_mask"].sum()) == 1
    assert torch.count_nonzero(batch["labels"] != IGNORE_INDEX) == 1
    assert batch["valid_action_token_ids"].tolist() == [[9]]
