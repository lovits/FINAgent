"""Offline scheduler-action batches produced by grouped environment rollouts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tradingagents.scheduler.actions import parse_action

from .collator import IGNORE_INDEX


class GRPORolloutDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.rows = self._load()

    def _load(self) -> list[dict[str, Any]]:
        rows = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    row["selected_action"] = parse_action(row["selected_action"]).value
                    row["valid_actions"] = [
                        parse_action(action).value for action in row["valid_actions"]
                    ]
                    for field in ("advantage", "old_logprob", "ref_logprob"):
                        row[field] = float(row[field])
                    if not isinstance(row["serialized_state"], str):
                        raise ValueError("serialized_state must be a string")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid GRPO row at {self.path}:{line_number}: {exc}"
                    ) from exc
                rows.append(row)
        if not rows:
            raise ValueError(f"GRPO rollout dataset is empty: {self.path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class GRPOCollator:
    def __init__(self, tokenizer: Any, *, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = []
        for row in rows:
            prefix = row["serialized_state"].rstrip() + "\n<SCHEDULER_ACTION>\n"
            prompt_ids = self.tokenizer(prefix, add_special_tokens=True)["input_ids"]
            action_ids = self.tokenizer(
                row["selected_action"], add_special_tokens=False
            )["input_ids"]
            if len(action_ids) != 1:
                raise ValueError("scheduler actions must be registered as single tokens")
            overflow = max(0, len(prompt_ids) + 1 - self.max_length)
            prompt_ids = prompt_ids[overflow:]
            input_ids = [*prompt_ids, action_ids[0]]
            action_mask = [0] * max(0, len(input_ids) - 2) + [1]
            valid_token_ids = []
            for action in row["valid_actions"]:
                ids = self.tokenizer(action, add_special_tokens=False)["input_ids"]
                if len(ids) != 1:
                    raise ValueError("valid scheduler actions must be single tokens")
                valid_token_ids.append(ids[0])
            if action_ids[0] not in valid_token_ids:
                raise ValueError("selected action is not included in valid_actions")
            encoded.append((row, input_ids, action_mask, valid_token_ids))

        width = max(len(input_ids) for _, input_ids, _, _ in encoded)
        mask_width = width - 1
        action_width = max(len(valid_ids) for _, _, _, valid_ids in encoded)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (width - len(values))

        return {
            "input_ids": torch.tensor(
                [
                    pad(input_ids, self.tokenizer.pad_token_id)
                    for _, input_ids, _, _ in encoded
                ]
            ),
            "attention_mask": torch.tensor(
                [pad([1] * len(input_ids), 0) for _, input_ids, _, _ in encoded]
            ),
            "action_mask": torch.tensor(
                [
                    mask + [0] * (mask_width - len(mask))
                    for _, _, mask, _ in encoded
                ],
                dtype=torch.bool,
            ),
            "prediction_indices": torch.tensor(
                [len(input_ids) - 2 for _, input_ids, _, _ in encoded]
            ),
            "selected_action_token_ids": torch.tensor(
                [input_ids[-1] for _, input_ids, _, _ in encoded]
            ),
            "valid_action_token_ids": torch.tensor(
                [
                    valid_ids + [0] * (action_width - len(valid_ids))
                    for _, _, _, valid_ids in encoded
                ]
            ),
            "valid_action_mask": torch.tensor(
                [
                    [1] * len(valid_ids) + [0] * (action_width - len(valid_ids))
                    for _, _, _, valid_ids in encoded
                ],
                dtype=torch.bool,
            ),
            "advantages": torch.tensor(
                [row["advantage"] for row, _, _, _ in encoded]
            ),
            "old_logprobs": torch.tensor(
                [row["old_logprob"] for row, _, _, _ in encoded]
            ),
            "ref_logprobs": torch.tensor(
                [row["ref_logprob"] for row, _, _, _ in encoded]
            ),
            "labels": torch.tensor(
                [
                    pad([IGNORE_INDEX] * (len(input_ids) - 1) + [input_ids[-1]], IGNORE_INDEX)
                    for _, input_ids, _, _ in encoded
                ]
            ),
        }
