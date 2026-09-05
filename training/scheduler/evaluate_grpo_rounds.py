"""Evaluate RL-1/RL-2/RL-3 on the same two held-out tasks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from tradingagents.scheduler.store import write_json_atomic
from .auto_review import AutomaticReviewer, automatic_reward
from .evaluate import evaluate_trajectories
from .generate import load_tasks
from .model import SchedulerModelConfig, load_scheduler_model
from .runtime_config import scheduler_runtime_config
from .sft_benchmark import BenchmarkPolicy, limit_expert_requests, run_one


def validate_tasks(training_tasks, evaluation_tasks):
    train_ids = {task["task_id"] for task in training_tasks}
    evaluation_ids = {task["task_id"] for task in evaluation_tasks}
    if len(evaluation_tasks) != 2 or len(evaluation_ids) != 2:
        raise ValueError("GRPO comparison requires exactly two distinct evaluation tasks")
    if train_ids & evaluation_ids:
        raise ValueError("GRPO evaluation tasks overlap its training tasks")


def summarize(values, reviews):
    summary = {}
    for label, trajectories in values.items():
        metrics = evaluate_trajectories(trajectories, {})
        for key in ("trader_match_rate", "mean_portfolio_rating_distance", "same_as_static_path_rate"):
            metrics.pop(key, None)
        reviewed = [reviews[t.trajectory_id] for t in trajectories
                    if reviews[t.trajectory_id]["status"] == "reviewed"]
        rewards = [automatic_reward(t, reviews[t.trajectory_id]).total for t in trajectories
                   if reviews[t.trajectory_id]["status"] != "unavailable"]
        metrics.update(
            quality_reviewed=len(reviewed),
            mean_quality=(sum(item["quality"] for item in reviewed) / len(reviewed)
                          if reviewed else None),
            mean_automatic_reward=(sum(rewards) / len(rewards) if rewards else None),
            task_results=[{"task_id": t.task_id, "status": t.execution_status,
                           "failure_reason": t.failure_reason} for t in trajectories],
        )
        summary[label] = metrics
    return summary


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    load_dotenv()
    plan = json.loads(Path(args.plan).read_text())
    training_tasks = load_tasks(plan["train_tasks"], split="train")
    evaluation_tasks = load_tasks(plan["eval_tasks"])
    validate_tasks(training_tasks, evaluation_tasks)
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO checkpoint evaluation requires CUDA")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["TRADINGAGENTS_OHLCV_SNAPSHOT_DIR"] = str(Path(plan["snapshot_dir"]).resolve())
    checkpoint_root = Path(plan["output_dir"])
    checkpoints = {f"rl-{number}": checkpoint_root / f"round-{number}/checkpoint"
                   for number in range(1, 4)}
    for checkpoint in checkpoints.values():
        if not (checkpoint / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"missing GRPO checkpoint: {checkpoint}")
    write_json_atomic(output / "manifest.json", {
        "models": {name: str(path) for name, path in checkpoints.items()},
        "tasks": evaluation_tasks,
        "planned_executions": 6,
        "comparison": "same two tasks across RL-1/RL-2/RL-3",
        "modes": ["learned"],
        "gpu_concurrency": 1,
        "expert_request_limit": 6,
    })
    torch.set_num_threads(2)
    gpu_lock = Lock()
    policies = {}
    for label, checkpoint in checkpoints.items():
        model, tokenizer, _ = load_scheduler_model(SchedulerModelConfig(
            base_model=plan["base_model"], base_revision=None,
            adapter_path=str(checkpoint)), training=False)
        model.config.use_cache = False
        model.requires_grad_(False)
        policies[label] = BenchmarkPolicy(
            model, tokenizer, policy_id=f"grpo-eval:{label}", device="cpu",
            gpu_lock=gpu_lock, max_context_tokens=32768,
        )
    runtime = scheduler_runtime_config({
        "llm_provider": "openrouter",
        "quick_think_llm": "z-ai/glm-5.3-flash",
        "deep_think_llm": "z-ai/glm-5.3-flash",
        "temperature": 0.0,
        "llm_max_retries": 1,
        "scheduler_max_steps": 16,
        "scheduler_max_context_tokens": 32768,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    })
    values = {label: [] for label in policies}
    write_json_atomic(output / "status.json", {"status": "running", "finished": 0, "planned": 6})
    with limit_expert_requests(6), ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(run_one, task, label, policy, runtime, output): label
                   for label, policy in policies.items() for task in evaluation_tasks}
        for future in as_completed(futures):
            values[futures[future]].append(future.result())
            write_json_atomic(output / "status.json", {
                "status": "running", "finished": sum(map(len, values.values())), "planned": 6})
    reviewer = AutomaticReviewer()
    reviews = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(reviewer.assess, trajectory): trajectory
                   for trajectories in values.values() for trajectory in trajectories}
        for future in as_completed(futures):
            trajectory = futures[future]
            reviews[trajectory.trajectory_id] = future.result()
            write_json_atomic(output / "quality-reviews.json", reviews)
    summary = summarize(values, reviews)
    write_json_atomic(output / "summary.json", summary)
    write_json_atomic(output / "status.json", {
        "status": "completed", "finished": 6, "planned": 6,
        "successful": sum(t.execution_status == "completed"
                          for trajectories in values.values() for t in trajectories),
    })


if __name__ == "__main__":
    main()
