"""Action-only SFT encoding and padding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

IGNORE_INDEX = -100


def encode_sft_example(
    tokenizer: Any,
    input_text: str,
    target_action: str,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Encode prompt as context and action/EOS as the only supervised tokens."""

    prefix = input_text.rstrip() + "\n<SCHEDULER_ACTION>\n"
    prompt_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    target_ids = tokenizer(target_action, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        target_ids = [*target_ids, eos_id]
    if not target_ids:
        raise ValueError("target action tokenized to an empty sequence")

    overflow = max(0, len(prompt_ids) + len(target_ids) - max_length)
    if overflow:
        prompt_ids = prompt_ids[overflow:]
    if len(target_ids) > max_length:
        raise ValueError("max_length is too small for the target action")
    input_ids = [*prompt_ids, *target_ids]
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class ActionOnlyCollator:
    def __init__(self, tokenizer: Any, *, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer requires pad_token_id or eos_token_id")
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = [
            encode_sft_example(
                self.tokenizer,
                str(feature["input_text"]),
                str(feature["target_action"]),
                max_length=self.max_length,
            )
            for feature in features
        ]
        width = max(len(item["input_ids"]) for item in encoded)

        def padded(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (width - len(values))

        return {
            "input_ids": torch.tensor(
                [padded(item["input_ids"], self.tokenizer.pad_token_id) for item in encoded]
            ),
            "attention_mask": torch.tensor(
                [padded(item["attention_mask"], 0) for item in encoded]
            ),
            "labels": torch.tensor(
                [padded(item["labels"], IGNORE_INDEX) for item in encoded]
            ),
        }
