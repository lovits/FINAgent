"""Convert accepted trajectories into action-only SFT examples."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tradingagents.scheduler.trajectory import SchedulerTrajectory

SFT_SCHEMA_VERSION = "v1"


@dataclass(frozen=True)
class SFTExample:
    input_text: str
    target_action: str
    metadata: dict[str, Any]
    schema_version: str = SFT_SCHEMA_VERSION


def examples_from_trajectory(trajectory: SchedulerTrajectory) -> list[SFTExample]:
    if trajectory.status != "accepted":
        raise ValueError("only accepted trajectories may become SFT examples")
    examples = []
    trajectory_metadata = trajectory.metadata
    for step in trajectory.steps:
        if step.error:
            raise ValueError("accepted trajectory contains an error step")
        examples.append(
            SFTExample(
                input_text=step.serialized_state,
                target_action=step.selected_action,
                metadata={
                    "trajectory_id": trajectory.trajectory_id,
                    "generation_key": trajectory_metadata.get("generation_key"),
                    "step_id": step.step_id,
                    "source": trajectory_metadata.get("source", trajectory.mode),
                    "task_id": trajectory.task_id,
                    "ticker": trajectory.ticker,
                    "trade_date": trajectory.trade_date,
                    "task_split": trajectory_metadata.get("task_split"),
                    "seed_family": trajectory_metadata.get("seed_family"),
                    "sector": trajectory_metadata.get("sector"),
                    "sample_index": trajectory_metadata.get("sample_index"),
                    "teacher_model": trajectory.teacher_model,
                    "teacher_prompt_version": trajectory.teacher_prompt_version,
                    "state_schema_version": trajectory.state_schema_version,
                    "action_schema_version": trajectory.action_schema_version,
                    "verifier_version": trajectory.verifier_version,
                },
            )
        )
    return examples


def examples_from_static_record(record: dict[str, Any]) -> list[SFTExample]:
    if record.get("record_type") != "static_langgraph":
        raise ValueError("record is not a Static LangGraph trajectory")
    if record.get("status") != "accepted":
        raise ValueError("only accepted Static trajectories may become SFT examples")
    task = record.get("task") or {}
    provenance = record.get("provenance") or {}
    examples = []
    for step in record.get("scheduler_examples") or []:
        if not step.get("action_valid"):
            raise ValueError("accepted Static trajectory contains an invalid action")
        examples.append(
            SFTExample(
                input_text=str(step["input_text"]),
                target_action=str(step["target_action"]),
                metadata={
                    "trajectory_id": record.get("trajectory_id"),
                    "generation_key": record.get("generation_key"),
                    "task_id": task.get("task_id"),
                    "ticker": task.get("ticker"),
                    "trade_date": task.get("trade_date"),
                    "task_split": task.get("split"),
                    "seed_family": task.get("seed_family"),
                    "sector": task.get("sector"),
                    "sample_index": record.get("sample_index"),
                    "step_id": step["step_id"],
                    "source": "original_static_langgraph",
                    "source_node": step.get("source_node"),
                    "dataset_version": record.get("dataset_version"),
                    "state_schema_version": record.get("state_schema_version"),
                    "action_schema_version": record.get("action_schema_version"),
                    "code_commit": provenance.get("code_commit"),
                    "information_cutoff": provenance.get("information_cutoff"),
                },
            )
        )
    return examples


def examples_from_jsonl(path: str | Path) -> Iterable[SFTExample]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("record_type") == "static_langgraph":
                    yield from examples_from_static_record(record)
                else:
                    trajectory = SchedulerTrajectory.from_dict(record)
                    yield from examples_from_trajectory(trajectory)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid trajectory at {path}:{line_number}: {exc}") from exc


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
    write_sft_jsonl(examples_from_jsonl(args.input), args.output)


if __name__ == "__main__":
    main()
