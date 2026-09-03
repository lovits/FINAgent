"""JSONL datasets used by scheduler SFT and offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tradingagents.scheduler.actions import parse_action


class SchedulerSFTDataset:
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
                    row["target_action"] = parse_action(row["target_action"]).value
                    if not isinstance(row["input_text"], str) or not row["input_text"]:
                        raise ValueError("input_text must be a non-empty string")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid SFT row at {self.path}:{line_number}: {exc}") from exc
                rows.append(row)
        if not rows:
            raise ValueError(f"SFT dataset is empty: {self.path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]
