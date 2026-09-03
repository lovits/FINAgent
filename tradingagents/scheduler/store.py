"""Append-only JSONL trajectory storage and atomic manifest writes."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .trajectory import SchedulerTrajectory


class TrajectoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._known_ids: set[str] | None = None

    def _load_ids(self) -> set[str]:
        if self._known_ids is not None:
            return self._known_ids
        self._known_ids = set()
        if not self.path.exists():
            return self._known_ids
        for line_number, record in enumerate(self.iter_dicts(), start=1):
            trajectory_id = record.get("trajectory_id")
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise ValueError(
                    f"missing trajectory_id at {self.path}:{line_number}"
                )
            if trajectory_id in self._known_ids:
                raise ValueError(f"duplicate trajectory_id in {self.path}: {trajectory_id}")
            self._known_ids.add(trajectory_id)
        return self._known_ids

    def contains(self, trajectory_id: str) -> bool:
        return trajectory_id in self._load_ids()

    def append(self, trajectory: SchedulerTrajectory) -> None:
        known = self._load_ids()
        if trajectory.trajectory_id in known:
            raise ValueError(f"trajectory already exists: {trajectory.trajectory_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trajectory.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        known.add(trajectory.trajectory_id)

    def iter_dicts(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {self.path}:{line_number}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object at {self.path}:{line_number}")
                yield value

    def load(self) -> list[SchedulerTrajectory]:
        return [SchedulerTrajectory.from_dict(value) for value in self.iter_dicts()]


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
