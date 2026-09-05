"""Run SFT epochs with held-out, paired real-harness evaluations between epochs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from tradingagents.reporting import write_report_tree
from tradingagents.scheduler.hf_policy import HFSchedulerPolicy
from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from tradingagents.scheduler.trajectory import SchedulerTrajectory
from tradingagents.scheduler.teacher_policy import OpenRouterTeacherGateway, TeacherSchedulerPolicy

from .audit import audit_trajectory
from .auto_review import AutomaticReviewer, automatic_reward
from .environment import TradingAgentsSchedulerEnvironment
from .evaluate import evaluate_sets, evaluate_trajectories
from .generate import load_tasks
from .profile import resolve_task_runtime
from .provenance import code_provenance, trajectory_provenance
from .runtime_config import scheduler_runtime_config
from .train_sft import SFTTrainConfig, train


class LockedPolicy:
    """Share one GPU model while independent expert tasks await API responses."""

    def __init__(self, policy, before_select=None):
        self.policy = policy
        self.policy_id = policy.policy_id
        self.lock = Lock()
        self.before_select = before_select

    def select_action(self, context):
        with self.lock:
            if self.before_select is not None:
                self.before_select(self.policy, context)
            return self.policy.select_action(context)


def validate_eval_tasks(tasks, train_path, validation_path):
    evaluation_ids = [t["task_id"] for t in tasks]
    if len(evaluation_ids) != 5 or len(set(evaluation_ids)) != 5:
        raise ValueError("real evaluation requires exactly 5 distinct tasks")
    excluded = set()
    for path in (train_path, validation_path):
        with Path(path).open() as handle:
            excluded.update(json.loads(line)["task_id"] for line in handle if line.strip())
    if excluded.intersection(evaluation_ids):
        raise ValueError("real evaluation tasks overlap SFT train/validation data")


def run_task(task, mode, policy, runtime, output, run_id):
    config, analysts = resolve_task_runtime(task, runtime)
    directory = output / task["task_id"] / mode
    try:
        result = TradingAgentsSchedulerEnvironment(config, selected_analysts=analysts).run(
            task,
            mode=mode,
            policy=policy if mode != "static" else None,
            run_id=run_id,
            trajectory_id=f"{run_id}:{task['task_id']}:{mode}",
        )
        trajectory = result.trajectory
        if trajectory.execution_status == "completed":
            write_report_tree(result.final_state, task["ticker"], directory / "report")
    except Exception as exc:
        policy_id = policy.policy_id if mode != "static" else "static-langgraph-v1"
        trajectory = SchedulerTrajectory(
            trajectory_id=f"{run_id}:{task['task_id']}:{mode}",
            run_id=run_id,
            task_id=task["task_id"],
            mode=mode,
            policy_id=policy_id,
            ticker=task["ticker"],
            trade_date=task["trade_date"],
            data_snapshot_id=task.get("data_snapshot_id"),
            execution_status="failed",
            failure_reason=f"initialization_or_report_error:{type(exc).__name__}",
            provenance=trajectory_provenance(
                config=config,
                task=task,
                mode=mode,
                policy_id=policy_id,
                selected_analysts=analysts,
                code=code_provenance(),
            ),
        )
    audit = audit_trajectory(trajectory)
    TrajectoryStore(directory / "raw.jsonl").append(trajectory)
    target = "accepted" if audit.audit_status in {"accepted", "warning"} else "rejected"
    TrajectoryStore(directory / f"{target}.jsonl").append(trajectory)
    return trajectory


def evaluate_epoch(epoch, model, tokenizer, checkpoint, *, tasks, output, max_length,
                   phase="full", include_teacher=False, stage_name=None, before_select=None,
                   include_static=True, policy_factory=None, existing_results=()):
    import torch

    stage = stage_name or (f"epoch-{epoch:02d}" + ("-half" if phase == "half" else ""))
    directory = output / stage
    directory.mkdir(parents=True, exist_ok=False)
    runtime = scheduler_runtime_config(
        {
            "llm_provider": "openrouter",
            "quick_think_llm": "z-ai/glm-5.3-flash",
            "deep_think_llm": "z-ai/glm-5.3-flash",
            "temperature": 0.0,
            "scheduler_max_steps": 16,
            "scheduler_max_context_tokens": max_length,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
        }
    )
    policy = LockedPolicy(
        (policy_factory or HFSchedulerPolicy)(
            model,
            tokenizer,
            policy_id=f"hf:{checkpoint}",
            device=str(next(model.parameters()).device),
            temperature=0.0,
            max_context_tokens=max_length,
        ),
        before_select=before_select,
    )
    policies = {"learned": policy}
    if include_static:
        policies["static"] = None
    if include_teacher:
        policies["teacher"] = TeacherSchedulerPolicy(OpenRouterTeacherGateway())
    values = {mode: [] for mode in policies}
    completed_keys = set()
    for trajectory in existing_results:
        key = (trajectory.task_id, trajectory.mode)
        if trajectory.mode not in policies or trajectory.task_id not in {t["task_id"] for t in tasks}:
            raise ValueError("reused evaluation result does not belong to this task group")
        if key in completed_keys:
            raise ValueError("duplicate reused evaluation result")
        completed_keys.add(key)
        values[trajectory.mode].append(trajectory)
    planned = len(tasks) * len(policies)
    write_json_atomic(
        directory / "progress.json", {"status": "running", "finished": len(completed_keys), "planned": planned}
    )
    with torch.random.fork_rng(), ThreadPoolExecutor(max_workers=planned) as executor:
        futures = [
            executor.submit(run_task, task, mode, mode_policy, runtime, directory, stage)
            for task in tasks
            for mode, mode_policy in policies.items()
            if (task["task_id"], mode) not in completed_keys
        ]
        for future in as_completed(futures):
            trajectory = future.result()
            values[trajectory.mode].append(trajectory)
            progress = {
                "status": "running",
                "finished": sum(map(len, values.values())),
                "planned": planned,
            }
            write_json_atomic(directory / "progress.json", progress)
            print(
                json.dumps(
                    {
                        "event": "scenario_finished",
                        "epoch": epoch,
                        "phase": phase,
                        "task_id": trajectory.task_id,
                        "mode": trajectory.mode,
                        "result": trajectory.execution_status,
                        **progress,
                    }
                ),
                flush=True,
            )
    if include_static:
        report = evaluate_sets(values["static"], **{k: v for k, v in values.items() if k != "static"})
    else:
        modes = {mode: evaluate_trajectories(items, {}) for mode, items in values.items()}
        for metrics in modes.values():
            for key in ("trader_match_rate", "mean_portfolio_rating_distance", "same_as_static_path_rate"):
                metrics.pop(key, None)
        report = {"evaluation_schema_version": "scheduler-evaluation-v1",
                  "baseline_comparison": None, "modes": modes}
    reviewer = AutomaticReviewer()
    quality = {mode: [] for mode in policies}
    for mode, trajectories in values.items():
        for trajectory in trajectories:
            review = reviewer.assess(trajectory)
            reward = (
                None
                if review["status"] == "unavailable"
                else automatic_reward(trajectory, review).to_dict()
            )
            quality[mode].append(
                {"task_id": trajectory.task_id, "review": review, "reward": reward}
            )
            write_json_atomic(directory / "quality_reviews.json", quality)
    report["quality_summary"] = {
        mode: {
            "reviewed": sum(item["review"]["status"] == "reviewed" for item in items),
            "unavailable": sum(item["review"]["status"] == "unavailable" for item in items),
            "mean_quality": _mean_quality(items),
        }
        for mode, items in quality.items()
    }
    report["failures"] = [
        {"task_id": t.task_id, "mode": t.mode, "reason": t.failure_reason}
        for group in values.values()
        for t in group
        if t.execution_status != "completed"
    ]
    report["interpretation"] = (
        "Development evaluation; outcome agreement is not financial correctness. "
        "Learned CPU inference and API-backed baselines do not have comparable scheduler hardware latency."
    )
    report["learned_device"] = str(next(model.parameters()).device)
    write_json_atomic(directory / "comparison.json", report)
    write_json_atomic(
        directory / "progress.json", {"status": "completed", "finished": planned, "planned": planned}
    )
    print(
        json.dumps({"event": "evaluation_completed", "epoch": epoch, "phase": phase,
                    "modes": report["modes"]}),
        flush=True,
    )
    torch.cuda.empty_cache()


def _mean_quality(items):
    scores = [item["review"]["quality"] for item in items if item["review"]["status"] == "reviewed"]
    return sum(scores) / len(scores) if scores else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--half-epoch-evaluation", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    config = SFTTrainConfig.from_json(args.config)
    tasks = load_tasks(args.tasks)
    validate_eval_tasks(tasks, config.train_path, config.validation_path)
    output = Path(config.output_dir)
    write_json_atomic(output / "cycle-config.json", {"training": asdict(config), "tasks": tasks})

    def callback(epoch, model, tokenizer, checkpoint, *, phase="full"):
        evaluate_epoch(
            epoch,
            model,
            tokenizer,
            checkpoint,
            tasks=tasks,
            output=output / "scenarios",
            max_length=config.max_length,
            phase=phase,
        )

    def half_callback(epoch, model, tokenizer, checkpoint):
        callback(epoch, model, tokenizer, checkpoint, phase="half")

    metrics = train(
        config, epoch_callback=callback,
        half_epoch_callback=half_callback if args.half_epoch_evaluation else None,
    )
    write_json_atomic(output / "cycle-complete.json", {"status": "completed", "metrics": metrics})
    print(json.dumps({"event": "cycle_completed", "metrics": metrics}), flush=True)


if __name__ == "__main__":
    main()
