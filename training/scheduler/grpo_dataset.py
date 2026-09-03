"""Validated GRPO rows and a collator aligned with masked scheduler actions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .sft_dataset import MaskedActionCollator


class SchedulerGRPODataset(Dataset):
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
                    if row["schema_version"] != "scheduler-grpo-v1":
                        raise ValueError("unsupported schema_version")
                    if row["selected_action"] not in row["valid_actions"]:
                        raise ValueError("selected_action is not valid")
                    for name in ("old_logprob", "ref_logprob", "advantage"):
                        row[name] = float(row[name])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid GRPO row at {self.path}:{line_number}: {exc}"
                    ) from exc
                rows.append(row)
        if not rows:
            raise ValueError(f"GRPO dataset is empty: {self.path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class GRPOCollator:
    def __init__(self, tokenizer: Any, *, max_length: int = 32768):
        self.action_collator = MaskedActionCollator(tokenizer, max_length=max_length)

    def __call__(self, rows):
        import torch

        action_rows = [
            {
                "sample_id": row["trajectory_id"],
                "input_text": row["serialized_state"],
                "valid_actions": row["valid_actions"],
                "target_action": row["selected_action"],
            }
            for row in rows
        ]
        batch = self.action_collator(action_rows)
        batch.update(
            {
                "old_logprobs": torch.tensor([row["old_logprob"] for row in rows]),
                "ref_logprobs": torch.tensor([row["ref_logprob"] for row in rows]),
                "advantages": torch.tensor([row["advantage"] for row in rows]),
            }
        )
        return batch
