"""Convert accepted trajectories into action-only SFT examples."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tradingagents.scheduler.trajectory import SchedulerTrajectory
from tradingagents.scheduler.trajectory_store import TrajectoryStore


@dataclass(frozen=True)
class SFTExample:
    input_text: str
    target_action: str
    metadata: dict[str, Any]


def examples_from_trajectory(trajectory: SchedulerTrajectory) -> list[SFTExample]:
    if trajectory.status != "accepted":
        raise ValueError("only accepted trajectories may become SFT examples")
    examples = []
    for step in trajectory.steps:
        if step.error:
            raise ValueError("accepted trajectory contains an error step")
        examples.append(
            SFTExample(
                input_text=step.serialized_state,
                target_action=step.selected_action,
                metadata={
                    "trajectory_id": trajectory.trajectory_id,
                    "step_id": step.step_id,
                    "source": trajectory.metadata.get("source", trajectory.mode),
                    "teacher_model": trajectory.teacher_model,
                    "teacher_prompt_version": trajectory.teacher_prompt_version,
                    "state_schema_version": trajectory.state_schema_version,
                    "action_schema_version": trajectory.action_schema_version,
                    "verifier_version": trajectory.verifier_version,
                },
            )
        )
    return examples


def write_sft_jsonl(examples: Iterable[SFTExample], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Accepted trajectory JSONL")
    parser.add_argument("--output", required=True, help="Destination SFT JSONL")
    args = parser.parse_args()
    examples = (
        example
        for trajectory in TrajectoryStore(args.input).load()
        for example in examples_from_trajectory(trajectory)
    )
    write_sft_jsonl(examples, args.output)


if __name__ == "__main__":
    main()
