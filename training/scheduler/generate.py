"""CLI and library entry point for one-trajectory-per-task data generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from tradingagents.scheduler.actions import ACTION_SCHEMA_VERSION
from tradingagents.scheduler.prompt import (
    ORCHESTRATION_PROFILE_VERSION,
    PROMPT_VERSION,
    STATE_SCHEMA_VERSION,
    TEACHER_PROMPT_VERSION,
)
from tradingagents.scheduler.registry import AGENT_CATALOG_VERSION
from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from tradingagents.scheduler.teacher_policy import (
    OpenRouterTeacherGateway,
    TeacherSchedulerPolicy,
)
from tradingagents.scheduler.trajectory import TRAJECTORY_SCHEMA_VERSION

from .audit import audit_trajectory
from .environment import TradingAgentsSchedulerEnvironment
from .manifest import summarize_trajectories
from .memory_snapshot import build_memory_snapshot
from .profile import resolve_task_runtime
from .provenance import code_provenance, expert_config_hash, generation_config_hash
from .runtime_config import scheduler_runtime_config

GENERATION_SCHEMA_VERSION = "scheduler-generation-v2"


def load_tasks(path: str | Path, *, split: str | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    tasks = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
                for field in ("task_id", "ticker", "trade_date"):
                    if not task.get(field):
                        raise ValueError(f"missing {field}")
                resolve_task_runtime(task, {})
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid task at {source}:{line_number}: {exc}") from exc
            if split is None or task.get("split") == split:
                tasks.append(task)
    return tasks


def generate_trajectories(
    tasks: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    mode: str,
    run_id: str,
    config: dict[str, Any],
    resume: bool = False,
    memory_log_path: str | None = None,
) -> dict[str, int]:
    if mode not in {"static", "teacher"}:
        raise ValueError("data generation mode must be static or teacher")
    started_at = datetime.now(UTC).isoformat()
    output = Path(output_dir)
    raw_store = TrajectoryStore(output / "raw.jsonl")
    accepted_store = TrajectoryStore(output / "accepted.jsonl")
    rejected_store = TrajectoryStore(output / "rejected.jsonl")
    policy = _teacher_policy(config) if mode == "teacher" else None
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
    task_count = 0
    task_dataset_versions = set()
    analyst_sets: set[tuple[str, ...]] = set()
    output_languages: set[str] = set()
    research_depths: set[str] = set()
    expert_config_hashes: set[str] = set()
    generation_config_hashes: set[str] = set()
    for source_task in tasks:
        task = dict(source_task)
        task_config, selected_analysts = resolve_task_runtime(task, config)
        analyst_sets.add(selected_analysts)
        output_languages.add(str(task["output_language"]))
        research_depths.add(str(task["research_depth"]))
        expert_config_hashes.add(
            expert_config_hash(task_config, selected_analysts=selected_analysts)
        )
        generation_config_hashes.add(
            generation_config_hash(
                task_config,
                mode=mode,
                policy_id=(
                    "static-langgraph-v1" if policy is None else policy.policy_id
                ),
                selected_analysts=selected_analysts,
            )
        )
        task_count += 1
        if task.get("dataset_version"):
            task_dataset_versions.add(str(task["dataset_version"]))
        if memory_log_path:
            snapshot = build_memory_snapshot(
                memory_log_path,
                ticker=str(task["ticker"]),
                trade_date=str(task["trade_date"]),
            )
            task["past_context"] = snapshot.context
            task["memory_snapshot_id"] = snapshot.snapshot_id
        identifier = _trajectory_id(task, mode, None if policy is None else policy.policy_id)
        if raw_store.contains(identifier):
            if not resume:
                raise FileExistsError(
                    f"trajectory already exists: {identifier}; pass --resume to skip it"
                )
            counts["skipped"] += 1
            continue
        environment = TradingAgentsSchedulerEnvironment(
            task_config,
            selected_analysts=selected_analysts,
        )
        result = environment.run(
            task,
            mode=mode,
            run_id=run_id,
            policy=policy,
            past_context=str(task.get("past_context") or ""),
            trajectory_id=identifier,
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

    code_metadata = code_provenance(config.get("project_dir"))
    write_json_atomic(
        output / "generation_manifest.json",
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "run_id": run_id,
            "mode": mode,
            "task_count": task_count,
            "task_dataset_versions": sorted(task_dataset_versions),
            "trajectories_per_task": 1,
            "task_inputs": {
                "analyst_sets": [list(values) for values in sorted(analyst_sets)],
                "output_languages": sorted(output_languages),
                "research_depths": sorted(research_depths),
            },
            "teacher_model": None if policy is None else policy.gateway.model,
            "teacher_prompt_version": (
                TEACHER_PROMPT_VERSION if policy is not None else None
            ),
            "orchestration_profile_version": ORCHESTRATION_PROFILE_VERSION,
            "expert_models": {
                "provider": config.get("llm_provider"),
                "quick": config.get("quick_think_llm"),
                "deep": config.get("deep_think_llm"),
            },
            "expert_config_hashes": sorted(expert_config_hashes),
            "schema_versions": {
                "trajectory": TRAJECTORY_SCHEMA_VERSION,
                "actions": ACTION_SCHEMA_VERSION,
                "state": STATE_SCHEMA_VERSION,
                "agent_catalog": AGENT_CATALOG_VERSION,
                "scheduler_prompt": PROMPT_VERSION,
            },
            **code_metadata,
            "generation_config_hashes": sorted(generation_config_hashes),
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "counts": counts,
            "dataset_counts": summarize_trajectories(raw_store.load()),
        },
    )
    return counts


def _teacher_policy(config: dict[str, Any]) -> TeacherSchedulerPolicy:
    return TeacherSchedulerPolicy(
        OpenRouterTeacherGateway(
            model=str(config.get("teacher_model", "z-ai/glm-5.3-flash")),
            base_url=str(config.get("teacher_base_url", "https://openrouter.ai/api/v1")),
            timeout_seconds=float(config.get("teacher_timeout_seconds", 90.0)),
            temperature=float(config.get("teacher_temperature", 0.2)),
            seed=config.get("teacher_seed"),
        )
    )


def _trajectory_id(
    task: dict[str, Any], mode: str, policy_id: str | None
) -> str:
    source = ":".join(
        (
            str(task["task_id"]),
            str(task.get("data_snapshot_id") or "none"),
            "+".join(task["selected_analysts"]),
            str(task["research_depth"]),
            str(task["output_language"]),
            mode,
            policy_id or "static-langgraph-v1",
        )
    )
    return f"{mode}-{hashlib.sha256(source.encode()).hexdigest()[:20]}"


def _runtime_config() -> dict[str, Any]:
    return scheduler_runtime_config()


def _select_task(tasks: list[dict[str, Any]], task_id: str | None) -> list[dict[str, Any]]:
    if task_id is None:
        return tasks
    selected = [task for task in tasks if task["task_id"] == task_id]
    if len(selected) != 1:
        raise ValueError(f"expected one task for {task_id}, found {len(selected)}")
    return selected


def _write_single_report(output_dir: str | Path, tasks: list[dict[str, Any]]) -> None:
    if len(tasks) != 1:
        raise ValueError("--write-report requires exactly one selected task")
    output = Path(output_dir)
    accepted = TrajectoryStore(output / "accepted.jsonl").load()
    if not accepted:
        return
    if len(accepted) != 1:
        raise ValueError(f"expected one accepted trajectory, found {len(accepted)}")
    from tradingagents.reporting import write_report_tree

    final_state = accepted[0].steps[-1].state_before
    write_report_tree(final_state, str(tasks[0]["ticker"]), output / "report")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("static", "teacher"), required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--memory-log")
    args = parser.parse_args()

    tasks = _select_task(load_tasks(args.tasks, split=args.split), args.task_id)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    run_id = args.run_id or datetime.now(UTC).strftime(f"{args.mode}-%Y%m%dT%H%M%SZ")
    counts = generate_trajectories(
        tasks,
        output_dir=args.output_dir,
        mode=args.mode,
        run_id=run_id,
        config=_runtime_config(),
        resume=args.resume,
        memory_log_path=args.memory_log,
    )
    if args.write_report:
        _write_single_report(args.output_dir, tasks)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
