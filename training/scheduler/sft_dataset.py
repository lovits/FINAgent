"""SFT dataset, hierarchical source sampler, and no-truncation collator."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Sampler

from .model import scheduler_input_ids


class SchedulerSFTDataset(Dataset):
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
                    if row["schema_version"] != "scheduler-sft-v1":
                        raise ValueError("unsupported schema_version")
                    if row["target_action"] not in row["valid_actions"]:
                        raise ValueError("target_action is not valid")
                    if row["source"] not in {"static", "teacher_verified"}:
                        raise ValueError("unknown source")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid SFT row at {self.path}:{line_number}: {exc}"
                    ) from exc
                rows.append(row)
        if not rows:
            raise ValueError(f"SFT dataset is empty: {self.path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def validate_task_isolation(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
) -> None:
    train_tasks = {str(row["task_id"]) for row in train_rows}
    validation_tasks = {str(row["task_id"]) for row in validation_rows}
    overlap = train_tasks & validation_tasks
    if overlap:
        raise ValueError(
            f"SFT Train/Validation task leakage detected: {len(overlap)} overlapping tasks"
        )


class HierarchicalSourceSampler(Sampler[int]):
    """Choose source, task, trajectory, then step to avoid long-path dominance."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        warmup_fraction: float = 0.1,
        warmup_teacher_probability: float = 0.5,
        teacher_probability: float = 0.8,
        total_epochs: int = 1,
        seed: int = 42,
    ):
        if not 0 <= warmup_fraction <= 1:
            raise ValueError("warmup_fraction must be between zero and one")
        if not 0 <= warmup_teacher_probability <= 1 or not 0 <= teacher_probability <= 1:
            raise ValueError("source probabilities must be between zero and one")
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        self.rows = rows
        self.warmup_fraction = warmup_fraction
        self.warmup_teacher_probability = warmup_teacher_probability
        self.teacher_probability = teacher_probability
        self.total_epochs = total_epochs
        self.seed = seed
        self.epoch = 0
        self.groups = self._group(rows)

    @staticmethod
    def _group(rows: Sequence[dict[str, Any]]):
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, row in enumerate(rows):
            groups[row["source"]][row["task_id"]][row["trajectory_id"]].append(index)
        return groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        warmup_draws = round(len(self) * self.total_epochs * self.warmup_fraction)
        for draw in range(len(self)):
            global_draw = self.epoch * len(self) + draw
            teacher_probability = (
                self.warmup_teacher_probability
                if global_draw < warmup_draws
                else self.teacher_probability
            )
            source = "teacher_verified" if rng.random() < teacher_probability else "static"
            if source not in self.groups:
                source = next(iter(self.groups))
            tasks = self.groups[source]
            task_id = rng.choice(tuple(tasks))
            trajectories = tasks[task_id]
            trajectory_id = rng.choice(tuple(trajectories))
            yield rng.choice(trajectories[trajectory_id])


class MaskedActionCollator:
    def __init__(self, tokenizer: Any, *, max_length: int = 32768):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = [self._encode(row) for row in rows]
        width = max(len(value["input_ids"]) for value in encoded)
        action_width = max(len(value["valid_token_ids"]) for value in encoded)

        def left_pad(values: list[int], fill: int) -> list[int]:
            # Training asks Qwen for only the final logit. Left padding keeps
            # that final position on the real generation prompt for every row.
            return [fill] * (width - len(values)) + values

        return {
            "input_ids": torch.tensor(
                [
                    left_pad(value["input_ids"], self.tokenizer.pad_token_id)
                    for value in encoded
                ]
            ),
            "attention_mask": torch.tensor(
                [left_pad([1] * len(value["input_ids"]), 0) for value in encoded]
            ),
            "prediction_indices": torch.full((len(encoded),), width - 1),
            "valid_action_token_ids": torch.tensor(
                [
                    value["valid_token_ids"]
                    + [0] * (action_width - len(value["valid_token_ids"]))
                    for value in encoded
                ]
            ),
            "valid_action_mask": torch.tensor(
                [
                    [1] * len(value["valid_token_ids"])
                    + [0] * (action_width - len(value["valid_token_ids"]))
                    for value in encoded
                ],
                dtype=torch.bool,
            ),
            "target_action_token_ids": torch.tensor(
                [value["target_token_id"] for value in encoded]
            ),
        }

    def _encode(self, row: dict[str, Any]) -> dict[str, Any]:
        input_ids = scheduler_input_ids(self.tokenizer, row["input_text"])
        if len(input_ids) > self.max_length:
            raise ValueError(
                f"SFT sample {row.get('sample_id')} has {len(input_ids)} tokens; "
                f"maximum is {self.max_length}"
            )
        valid_token_ids = [self._action_token_id(action) for action in row["valid_actions"]]
        target_token_id = self._action_token_id(row["target_action"])
        if target_token_id not in valid_token_ids:
            raise ValueError("target action is not included in valid actions")
        return {
            "input_ids": list(input_ids),
            "valid_token_ids": valid_token_ids,
            "target_token_id": target_token_id,
        }

    def _action_token_id(self, action: str) -> int:
        values = self.tokenizer(action, add_special_tokens=False)["input_ids"]
        if len(values) != 1:
            raise ValueError(f"scheduler action is not one token: {action}")
        return int(values[0])
