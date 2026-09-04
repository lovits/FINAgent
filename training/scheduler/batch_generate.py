"""Run a collection plan concurrently in isolated child processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tradingagents.scheduler.store import write_json_atomic

from .generate import load_tasks

BATCH_MANIFEST_VERSION = "scheduler-batch-v1"
PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


@dataclass(frozen=True)
class BatchJob:
    task_id: str
    mode: str
    run_id: str
    output_dir: str


def build_jobs(
    tasks: list[dict[str, Any]], *, output_root: str | Path, run_prefix: str
) -> list[BatchJob]:
    jobs = []
    root = Path(output_root)
    for index, task in enumerate(tasks, start=1):
        role = task.get("collection_role")
        if role not in {"paired", "teacher_only"}:
            raise ValueError(f"invalid collection_role for {task['task_id']}: {role}")
        label = _task_label(index, task)
        modes = ("static", "teacher") if role == "paired" else ("teacher",)
        section = "paired" if role == "paired" else "teacher-only"
        for mode in modes:
            jobs.append(
                BatchJob(
                    task_id=str(task["task_id"]),
                    mode=mode,
                    run_id=f"{run_prefix}-{index:02d}-{mode}",
                    output_dir=str(root / section / label / mode),
                )
            )
    return jobs


def unstarted_jobs(jobs: list[BatchJob]) -> list[BatchJob]:
    return [job for job in jobs if not Path(job.output_dir).exists()]


def _task_label(index: int, task: dict[str, Any]) -> str:
    family = str(task.get("seed_family", "task")).replace("_", "-")
    return f"{index:02d}-{str(task['ticker']).lower()}-{family}"


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in PROXY_ENV_NAMES:
        environment.pop(name, None)
    return environment


def _terminal_result(job: BatchJob) -> dict[str, Any] | None:
    output = Path(job.output_dir)
    if (output / "accepted.jsonl").exists():
        return {**asdict(job), "status": "accepted", "returncode": 0}
    if (output / "rejected.jsonl").exists():
        return {**asdict(job), "status": "rejected", "returncode": 0}
    return None


def run_job(job: BatchJob, *, repo: Path, tasks_path: Path) -> dict[str, Any]:
    existing = _terminal_result(job)
    if existing is not None:
        return existing
    output = Path(job.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = _generation_command(job, tasks_path)
    with (output / "job.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=repo,
            env=_clean_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    terminal = _terminal_result(job)
    if terminal is not None:
        return {**terminal, "returncode": result.returncode}
    return {**asdict(job), "status": "failed", "returncode": result.returncode}


def _generation_command(job: BatchJob, tasks_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "training.scheduler.generate",
        "--tasks",
        str(tasks_path),
        "--task-id",
        job.task_id,
        "--mode",
        job.mode,
        "--run-id",
        job.run_id,
        "--output-dir",
        job.output_dir,
        "--resume",
        "--write-report",
    ]


def run_batch(
    jobs: list[BatchJob],
    *,
    repo: Path,
    tasks_path: Path,
    output_root: Path,
    max_workers: int,
) -> dict[str, Any]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    results = []
    _write_batch_manifest(output_root, jobs, results, status="running")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_job, job, repo=repo, tasks_path=tasks_path): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    **asdict(job),
                    "status": "failed",
                    "returncode": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            _write_batch_manifest(output_root, jobs, results, status="running")
    final_status = "completed" if all(r["status"] != "failed" for r in results) else "failed"
    return _write_batch_manifest(output_root, jobs, results, status=final_status)


def _write_batch_manifest(
    output_root: Path,
    jobs: list[BatchJob],
    results: list[dict[str, Any]],
    *,
    status: str,
) -> dict[str, Any]:
    status_counts = Counter(str(result["status"]) for result in results)
    manifest = {
        "schema_version": BATCH_MANIFEST_VERSION,
        "status": status,
        "planned_jobs": len(jobs),
        "finished_jobs": len(results),
        "pending_jobs": len(jobs) - len(results),
        "status_counts": dict(status_counts),
        "results": sorted(results, key=lambda value: (value["task_id"], value["mode"])),
    }
    write_json_atomic(output_root / "batch_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--only-unstarted", action="store_true")
    parser.add_argument("--manifest-dir")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    tasks_path = Path(args.tasks).resolve()
    output_root = Path(args.output_dir).resolve()
    jobs = build_jobs(
        load_tasks(tasks_path, split="train"),
        output_root=output_root,
        run_prefix=args.run_prefix,
    )
    if args.only_unstarted:
        jobs = unstarted_jobs(jobs)
    manifest_root = (
        Path(args.manifest_dir).resolve() if args.manifest_dir else output_root
    )
    manifest = run_batch(
        jobs,
        repo=repo,
        tasks_path=tasks_path,
        output_root=manifest_root,
        max_workers=args.max_workers,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
