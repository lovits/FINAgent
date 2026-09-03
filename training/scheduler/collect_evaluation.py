"""Collect one deterministic learned-policy trajectory per evaluation task."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from tradingagents.scheduler.hf_policy import load_hf_scheduler_policy
from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic

from .audit import audit_trajectory
from .environment import TradingAgentsSchedulerEnvironment
from .generate import load_tasks
from .provenance import code_provenance, expert_config_hash, generation_config_hash
from .runtime_config import scheduler_runtime_config

EVALUATION_COLLECTION_SCHEMA_VERSION = "scheduler-evaluation-collection-v1"


@dataclass(frozen=True)
class EvaluationCollectionConfig:
    tasks_path: str
    adapter_path: str
    output_dir: str
    run_id: str
    base_model: str = "Qwen/Qwen3-1.7B"
    base_revision: str | None = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    split: str = "validation"
    max_context_tokens: int = 32768
    max_steps: int = 16
    selected_analysts: tuple[str, ...] = (
        "market",
        "social",
        "news",
        "fundamentals",
    )
    limit: int | None = None
    resume: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> EvaluationCollectionConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if "selected_analysts" in value:
            value["selected_analysts"] = tuple(value["selected_analysts"])
        return cls(**value)


def collect(config: EvaluationCollectionConfig) -> dict[str, int]:
    started_at = datetime.now(UTC).isoformat()
    tasks = load_tasks(config.tasks_path, split=config.split)
    if config.limit is not None:
        tasks = tasks[: config.limit]
    runtime_config = scheduler_runtime_config(
        {
            "orchestration_mode": "learned",
            "scheduler_base_model": config.base_model,
            "scheduler_base_revision": config.base_revision,
            "scheduler_adapter_path": config.adapter_path,
            "scheduler_max_context_tokens": config.max_context_tokens,
            "scheduler_max_steps": config.max_steps,
            "scheduler_action_temperature": 0.0,
            "temperature": 0.0,
        }
    )
    policy = load_hf_scheduler_policy(runtime_config)
    environment = TradingAgentsSchedulerEnvironment(
        runtime_config,
        selected_analysts=config.selected_analysts,
    )
    output = Path(config.output_dir)
    raw_store = TrajectoryStore(output / "raw.jsonl")
    accepted_store = TrajectoryStore(output / "accepted.jsonl")
    rejected_store = TrajectoryStore(output / "rejected.jsonl")
    counts = {"accepted": 0, "rejected": 0, "skipped": 0}

    for task in tasks:
        trajectory_id = _trajectory_id(task, config.run_id, policy.policy_id)
        if raw_store.contains(trajectory_id):
            if not config.resume:
                raise FileExistsError(
                    f"trajectory already exists: {trajectory_id}; enable resume to skip it"
                )
            counts["skipped"] += 1
            continue
        result = environment.run(
            task,
            mode="learned",
            run_id=config.run_id,
            policy=policy,
            trajectory_id=trajectory_id,
        )
        audit = audit_trajectory(result.trajectory)
        raw_store.append(result.trajectory)
        if audit.audit_status in {"accepted", "warning"}:
            accepted_store.append(result.trajectory)
            counts["accepted"] += 1
        else:
            rejected_store.append(result.trajectory)
            counts["rejected"] += 1

    code_metadata = code_provenance(runtime_config.get("project_dir"))
    write_json_atomic(
        output / "collection_manifest.json",
        {
            "schema_version": EVALUATION_COLLECTION_SCHEMA_VERSION,
            "config": asdict(config),
            "policy_id": policy.policy_id,
            "task_count": len(tasks),
            "task_key_fields": ["task_id", "data_snapshot_id"],
            "scheduler_temperature": 0.0,
            "expert_temperature": 0.0,
            "expert_config_hash": expert_config_hash(
                runtime_config,
                selected_analysts=config.selected_analysts,
            ),
            **code_metadata,
            "generation_config_hash": generation_config_hash(
                runtime_config,
                mode="learned",
                policy_id=policy.policy_id,
                selected_analysts=config.selected_analysts,
            ),
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "counts": counts,
        },
    )
    return counts


def _trajectory_id(task: dict, run_id: str, policy_id: str) -> str:
    value = ":".join(
        (
            run_id,
            str(task["task_id"]),
            str(task.get("data_snapshot_id") or "none"),
            policy_id,
        )
    )
    return f"learned-eval-{hashlib.sha256(value.encode()).hexdigest()[:20]}"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = collect(EvaluationCollectionConfig.from_json(args.config))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
