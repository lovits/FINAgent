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
from .manifest import summarize_trajectories
from .profile import resolve_task_runtime, validate_shallow_runtime
from .provenance import code_provenance
from .runtime_config import scheduler_runtime_config

EVALUATION_COLLECTION_SCHEMA_VERSION = "scheduler-evaluation-collection-v2"


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
    limit: int | None = None
    resume: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> EvaluationCollectionConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
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
    validate_shallow_runtime(runtime_config)
    policy = load_hf_scheduler_policy(runtime_config)
    output = Path(config.output_dir)
    raw_store = TrajectoryStore(output / "raw.jsonl")
    accepted_store = TrajectoryStore(output / "accepted.jsonl")
    rejected_store = TrajectoryStore(output / "rejected.jsonl")
    counts = {
        "accepted": 0,
        "rejected": 0,
        "skipped": 0,
        "completed": 0,
        "failed": 0,
        "fallback": 0,
        "budget_exhausted": 0,
        "context_overflow": 0,
    }

    for task in tasks:
        task_config, selected_analysts = resolve_task_runtime(task, runtime_config)
        trajectory_id = _trajectory_id(task, config.run_id, policy.policy_id)
        if raw_store.contains(trajectory_id):
            if not config.resume:
                raise FileExistsError(
                    f"trajectory already exists: {trajectory_id}; enable resume to skip it"
                )
            counts["skipped"] += 1
            continue
        environment = TradingAgentsSchedulerEnvironment(
            task_config,
            selected_analysts=selected_analysts,
        )
        result = environment.run(
            task,
            mode="learned",
            run_id=config.run_id,
            policy=policy,
            trajectory_id=trajectory_id,
        )
        counts[result.trajectory.execution_status] += 1
        audit = audit_trajectory(result.trajectory)
        raw_store.append(result.trajectory)
        if audit.audit_status in {"accepted", "warning"}:
            accepted_store.append(result.trajectory)
            counts["accepted"] += 1
        else:
            rejected_store.append(result.trajectory)
            counts["rejected"] += 1

    code_metadata = code_provenance(runtime_config.get("project_dir"))
    collected = raw_store.load()
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
            "task_inputs": {
                "analyst_sets": sorted(
                    {
                        tuple(trajectory.provenance.get("selected_analysts") or ())
                        for trajectory in collected
                    }
                ),
                "output_languages": sorted(
                    {
                        str(trajectory.provenance["output_language"])
                        for trajectory in collected
                    }
                ),
                "research_depths": sorted(
                    {
                        str(trajectory.provenance["research_depth"])
                        for trajectory in collected
                    }
                ),
            },
            "expert_config_hashes": sorted(
                {
                    str(trajectory.provenance["expert_config_hash"])
                    for trajectory in collected
                }
            ),
            **code_metadata,
            "generation_config_hashes": sorted(
                {
                    str(trajectory.provenance["generation_config_hash"])
                    for trajectory in collected
                }
            ),
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "counts": counts,
            "dataset_counts": summarize_trajectories(collected),
        },
    )
    return counts


def _trajectory_id(task: dict, run_id: str, policy_id: str) -> str:
    value = ":".join(
        (
            run_id,
            str(task["task_id"]),
            str(task.get("data_snapshot_id") or "none"),
            "+".join(task["selected_analysts"]),
            str(task["research_depth"]),
            str(task["output_language"]),
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
