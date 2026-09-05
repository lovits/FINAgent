"""Validated GRPO rows and a collator aligned with masked scheduler actions."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .sft_dataset import MaskedActionCollator


class SchedulerGRPODataset(Dataset):
    def __init__(self, path: str | Path, *, expected_group_size: int | None = None):
        if expected_group_size is not None and expected_group_size < 2:
            raise ValueError("expected_group_size must be at least two")
        self.path = Path(path)
        self.rows = self._load()
        self._validate_groups(expected_group_size)
        self._assign_loss_weights()

    def _assign_loss_weights(self) -> None:
        groups = defaultdict(lambda: defaultdict(list))
        for row in self.rows:
            groups[row["rollout_group_id"]][row["trajectory_id"]].append(row)
        for trajectories in groups.values():
            for rows in trajectories.values():
                # Uniform row sampling now estimates mean(group, trajectory, step).
                weight = len(self.rows) / (len(groups) * len(trajectories) * len(rows))
                for row in rows:
                    row["loss_weight"] = weight

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
                    for name in (
                        "old_logprob",
                        "ref_logprob",
                        "reward_total",
                        "advantage",
                    ):
                        row[name] = float(row[name])
                        if not math.isfinite(row[name]):
                            raise ValueError(f"{name} must be finite")
                    if not row.get("rollout_group_id") or not row.get("trajectory_id"):
                        raise ValueError("rollout group and trajectory ids are required")
                    if not row.get("task_id"):
                        raise ValueError("task_id is required")
                    if not isinstance(row.get("step_id"), int) or row["step_id"] < 0:
                        raise ValueError("step_id must be a non-negative integer")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid GRPO row at {self.path}:{line_number}: {exc}"
                    ) from exc
                rows.append(row)
        if not rows:
            raise ValueError(f"GRPO dataset is empty: {self.path}")
        return rows

    def _validate_groups(self, expected_group_size: int | None) -> None:
        groups = defaultdict(lambda: defaultdict(list))
        group_tasks = defaultdict(set)
        for row in self.rows:
            group_id = row["rollout_group_id"]
            groups[group_id][row["trajectory_id"]].append(row)
            group_tasks[group_id].add(row["task_id"])

        for group_id, trajectories in groups.items():
            if len(group_tasks[group_id]) != 1:
                raise ValueError(f"GRPO group {group_id} mixes multiple tasks")
            if expected_group_size is not None and len(trajectories) != expected_group_size:
                raise ValueError(
                    f"GRPO group {group_id} has {len(trajectories)} trajectories; "
                    f"expected {expected_group_size}"
                )
            advantages = []
            for trajectory_id, rows in trajectories.items():
                step_ids = sorted(row["step_id"] for row in rows)
                if step_ids != list(range(len(step_ids))):
                    raise ValueError(f"GRPO trajectory {trajectory_id} has non-contiguous steps")
                trajectory_advantages = {row["advantage"] for row in rows}
                trajectory_rewards = {row["reward_total"] for row in rows}
                if len(trajectory_advantages) != 1 or len(trajectory_rewards) != 1:
                    raise ValueError(
                        f"GRPO trajectory {trajectory_id} has inconsistent credit values"
                    )
                advantages.append(next(iter(trajectory_advantages)))
            if len(advantages) >= 2 and not math.isclose(
                sum(advantages) / len(advantages),
                0.0,
                abs_tol=1e-5,
            ):
                raise ValueError(f"GRPO group {group_id} advantages are not zero-mean")

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
                "loss_weights": torch.tensor([row.get("loss_weight", 1.0) for row in rows]),
            }
        )
        return batch
