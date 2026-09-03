"""Append-only JSONL storage for scheduler trajectories."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from .trajectory import SchedulerTrajectory


class TrajectoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, trajectory: SchedulerTrajectory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            trajectory.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "ab") as handle:
            handle.write(line.encode("utf-8"))

    def load(self) -> Iterator[SchedulerTrajectory]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield SchedulerTrajectory.from_dict(json.loads(line))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid trajectory at {self.path}:{line_number}: {exc}"
                    ) from exc
