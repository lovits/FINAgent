"""Generate verified Static-Teacher or Strong-Teacher scheduler trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.scheduler.teacher_gateway import OpenRouterTeacherGateway
from tradingagents.scheduler.teacher_policy import StrongTeacherPolicy
from tradingagents.scheduler.trajectory_store import TrajectoryStore
from tradingagents.scheduler.verifier import verify_trajectory

from .environment import TradingAgentsRolloutEnvironment
from .reward import parse_trader_action, portfolio_similarity, score_trajectory
from .static_trace import StaticLangGraphRunner, append_static_record

GENERATION_SCHEMA_VERSION = "v1"


def _code_revision() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if head.returncode != 0:
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        check=False,
        text=True,
    )
    suffix = "-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else ""
    return head.stdout.strip() + suffix


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
                for field in ("ticker", "trade_date"):
                    if not task.get(field):
                        raise ValueError(f"missing {field}")
                task.setdefault(
                    "task_id",
                    f"{task['ticker']}:{task['trade_date']}:{task.get('asset_type', 'stock')}",
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid task at {path}:{line_number}: {exc}") from exc
            tasks.append(task)
    if not tasks:
        raise ValueError(f"task file is empty: {path}")
    return tasks


def select_generation_tasks(
    tasks: list[dict[str, Any]],
    *,
    dataset_split: str | None = None,
    tasks_per_family: int | None = None,
) -> list[dict[str, Any]]:
    selected = [task for task in tasks if dataset_split is None or task.get("split") == dataset_split]
    if not selected:
        raise ValueError(f"no tasks match split {dataset_split!r}")
    if tasks_per_family is None:
        return selected
    if tasks_per_family < 1:
        raise ValueError("tasks_per_family must be positive")

    families = tuple(dict.fromkeys(task.get("seed_family") for task in selected))
    if None in families:
        raise ValueError("balanced task selection requires seed_family on every task")
    counts = dict.fromkeys(families, 0)
    balanced = []
    for task in selected:
        family = task["seed_family"]
        if counts[family] < tasks_per_family:
            balanced.append(task)
            counts[family] += 1
    incomplete = {family: count for family, count in counts.items() if count < tasks_per_family}
    if incomplete:
        raise ValueError(f"not enough tasks for balanced generation: {incomplete}")
    return balanced


def generation_key(
    task: dict[str, Any],
    policy_kind: str,
    sample_index: int,
    generation_signature: str,
) -> str:
    return ":".join(
        (
            str(task["task_id"]),
            policy_kind,
            str(sample_index),
            GENERATION_SCHEMA_VERSION,
            generation_signature,
        )
    )


def build_generation_signature(
    config: dict[str, Any],
    policy_kind: str,
    selected_analysts: tuple[str, ...],
) -> str:
    payload = {
        "generation_schema_version": GENERATION_SCHEMA_VERSION,
        "code_revision": _code_revision(),
        "policy": policy_kind,
        "selected_analysts": selected_analysts,
        "llm_provider": config.get("llm_provider"),
        "quick_model": config.get("quick_think_llm"),
        "deep_model": config.get("deep_think_llm"),
        "teacher_model": config.get("teacher_model") if policy_kind == "teacher" else None,
        "max_debate_rounds": config.get("max_debate_rounds"),
        "max_risk_rounds": config.get("max_risk_discuss_rounds"),
        "temperature": config.get("temperature"),
        "data_vendors": config.get("data_vendors"),
        "tool_vendors": config.get("tool_vendors"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _existing_generation_keys(
    output: Path, expected_signature: str
) -> set[str]:
    keys = set()
    for name in ("accepted.jsonl", "rejected.jsonl"):
        path = output / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = record.get("generation_key") or (record.get("metadata") or {}).get(
                        "generation_key"
                    )
                    signature = record.get("generation_signature") or (
                        record.get("metadata") or {}
                    ).get("generation_signature")
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid existing trajectory at {path}:{line_number}") from exc
                if not key:
                    raise ValueError(
                        f"existing trajectory has no generation_key at {path}:{line_number}; "
                        "choose a new output directory"
                    )
                if signature != expected_signature:
                    raise ValueError(
                        f"generation configuration differs at {path}:{line_number}; "
                        "choose a new output directory"
                    )
                keys.add(str(key))
    return keys


def _stamp_static_record(
    record: dict[str, Any],
    *,
    generation_key_value: str,
    generation_signature: str,
    sample_index: int,
) -> None:
    record["generation_schema_version"] = GENERATION_SCHEMA_VERSION
    record["generation_key"] = generation_key_value
    record["generation_signature"] = generation_signature
    record["trajectory_id"] = str(uuid5(NAMESPACE_URL, generation_key_value))
    record["sample_index"] = sample_index


def write_generation_manifest(
    *,
    output_dir: str | Path,
    task_path: str | Path,
    tasks: list[dict[str, Any]],
    policy_kind: str,
    trajectories_per_task: int,
    selected_analysts: tuple[str, ...],
    resume: bool,
    counts: dict[str, int],
    generation_signature: str,
) -> None:
    source = Path(task_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_counts = {}
    for status in ("accepted", "rejected"):
        path = output / f"{status}.jsonl"
        output_counts[status] = (
            sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            if path.exists()
            else 0
        )
    manifest = {
        "generation_schema_version": GENERATION_SCHEMA_VERSION,
        "updated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "task_source": {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "selected_tasks": len(tasks),
            "split_counts": dict(Counter(task.get("split") for task in tasks)),
            "family_counts": dict(Counter(task.get("seed_family") for task in tasks)),
        },
        "policy": policy_kind,
        "generation_signature": generation_signature,
        "trajectories_per_task": trajectories_per_task,
        "selected_analysts": list(selected_analysts),
        "resume": resume,
        "last_run_counts": counts,
        "output_record_counts": output_counts,
        "secrets_recorded": False,
    }
    (output / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _quality_ok(
    final_state: dict[str, Any],
    static_state: dict[str, Any],
    *,
    min_rating_similarity: float,
) -> bool:
    if portfolio_similarity(
        str(final_state.get("final_trade_decision") or ""),
        str(static_state.get("final_trade_decision") or ""),
    ) < min_rating_similarity:
        return False
    candidate = parse_trader_action(final_state.get("trader_investment_plan"))
    reference = parse_trader_action(static_state.get("trader_investment_plan"))
    return candidate is not None and candidate == reference


def generate(
    *,
    tasks: list[dict[str, Any]],
    output_dir: str | Path,
    policy_kind: str,
    trajectories_per_task: int,
    selected_analysts: tuple[str, ...],
    min_rating_similarity: float,
    config: dict[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, int]:
    runtime_config = config or DEFAULT_CONFIG
    signature = build_generation_signature(
        runtime_config,
        policy_kind,
        selected_analysts,
    )
    output = Path(output_dir)
    existing_keys = _existing_generation_keys(output, signature)
    if existing_keys and not resume:
        raise FileExistsError(
            f"trajectory output already contains {len(existing_keys)} generated records; "
            "use resume=True or choose a new output directory"
        )
    counts = {"accepted": 0, "rejected": 0, "skipped": 0}
    if policy_kind == "static":
        runner = StaticLangGraphRunner(
            runtime_config,
            selected_analysts=selected_analysts,
        )
        for task in tasks:
            for sample_index in range(trajectories_per_task):
                key = generation_key(task, policy_kind, sample_index, signature)
                if key in existing_keys:
                    counts["skipped"] += 1
                    continue
                record = runner.run(task)
                _stamp_static_record(
                    record,
                    generation_key_value=key,
                    generation_signature=signature,
                    sample_index=sample_index,
                )
                status = record["status"]
                append_static_record(record, output / f"{status}.jsonl")
                counts[status] += 1
        return counts
    if policy_kind != "teacher":
        raise ValueError("policy_kind must be 'static' or 'teacher'")

    policy = StrongTeacherPolicy(
        OpenRouterTeacherGateway(model=runtime_config["teacher_model"])
    )
    source = "strong_teacher_verified"

    accepted_store = TrajectoryStore(output / "accepted.jsonl")
    rejected_store = TrajectoryStore(output / "rejected.jsonl")
    environment = TradingAgentsRolloutEnvironment(
        runtime_config,
        selected_analysts=selected_analysts,
    )
    for task_index, task in enumerate(tasks):
        for sample_index in range(trajectories_per_task):
            key = generation_key(task, policy_kind, sample_index, signature)
            if key in existing_keys:
                counts["skipped"] += 1
                continue
            result = environment.run(
                task,
                policy,
                seed=task_index * trajectories_per_task + sample_index,
            )
            quality_ok = _quality_ok(
                result.final_state,
                result.static_state,
                min_rating_similarity=min_rating_similarity,
            )
            verification = verify_trajectory(
                result.trajectory,
                result.final_state,
                quality_ok=quality_ok,
            )
            trajectory = result.trajectory
            trajectory.metadata.update(
                {
                    "source": source,
                    "generation_schema_version": GENERATION_SCHEMA_VERSION,
                    "generation_key": key,
                    "generation_signature": signature,
                    "sample_index": sample_index,
                    "task_split": task.get("split"),
                    "seed_family": task.get("seed_family"),
                    "sector": task.get("sector"),
                    "information_cutoff": task.get("trade_date"),
                }
            )
            trajectory.verifier_version = verification.verifier_version
            trajectory.reward = score_trajectory(
                trajectory, result.final_state, result.static_state
            ).to_dict()
            if verification.accepted:
                trajectory.status = "accepted"
                trajectory.failure_reason = None
                accepted_store.append(trajectory)
                counts["accepted"] += 1
            else:
                trajectory.status = "rejected"
                trajectory.failure_reason = verification.reason
                rejected_store.append(trajectory)
                counts["rejected"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", choices=("static", "teacher"), required=True)
    parser.add_argument("--trajectories-per-task", type=int, default=None)
    parser.add_argument(
        "--analysts", default="market,social,news,fundamentals"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-rating-similarity", type=float, default=0.5)
    parser.add_argument("--task-split", help="Optional dataset split, e.g. train")
    parser.add_argument(
        "--tasks-per-family",
        type=int,
        help="Optional balanced cap within the selected split",
    )
    args = parser.parse_args()
    trajectories = args.trajectories_per_task or (1 if args.policy == "static" else 2)
    tasks = select_generation_tasks(
        load_tasks(args.tasks),
        dataset_split=args.task_split,
        tasks_per_family=args.tasks_per_family,
    )
    selected_analysts = tuple(
        part.strip() for part in args.analysts.split(",") if part.strip()
    )
    counts = generate(
        tasks=tasks,
        output_dir=args.output_dir,
        policy_kind=args.policy,
        trajectories_per_task=trajectories,
        selected_analysts=selected_analysts,
        min_rating_similarity=args.min_rating_similarity,
        resume=args.resume,
    )
    write_generation_manifest(
        output_dir=args.output_dir,
        task_path=args.tasks,
        tasks=tasks,
        policy_kind=args.policy,
        trajectories_per_task=trajectories,
        selected_analysts=selected_analysts,
        resume=args.resume,
        counts=counts,
        generation_signature=build_generation_signature(
            DEFAULT_CONFIG,
            args.policy,
            selected_analysts,
        ),
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
