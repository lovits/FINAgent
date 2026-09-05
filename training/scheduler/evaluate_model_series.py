"""Evaluate RL checkpoints on the exact seven-task SFT benchmark."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from threading import BoundedSemaphore, Lock
import time

from dotenv import load_dotenv

from tradingagents.scheduler.hf_policy import HFSchedulerPolicy
from tradingagents.scheduler.store import write_json_atomic
from .auto_review import AutomaticReviewer
from .generate import load_tasks
from .model import SchedulerModelConfig, load_scheduler_model
from .runtime_config import scheduler_runtime_config
from .sft_benchmark import (
    limit_expert_requests, prepare_resume, run_one, summarize, validate_tasks,
)


class ConcurrentCheckpointPolicy(HFSchedulerPolicy):
    """Allow several distinct checkpoints on GPU while serializing each model."""

    def __init__(self, *args, gpu_slots, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_slots = gpu_slots
        self.model_lock = Lock()
        self.timings = defaultdict(lambda: {
            "queue_seconds": 0.0, "gpu_service_seconds": 0.0,
        })

    def _distribution(self, context):
        import torch

        requested = time.monotonic()
        with self.model_lock, self.gpu_slots:
            started = time.monotonic()
            timing = self.timings[context.task_id]
            timing["queue_seconds"] += started - requested
            try:
                self.model.to("cuda")
                self.device = "cuda"
                return super()._distribution(context).cpu()
            finally:
                self.model.to("cpu")
                self.device = "cpu"
                torch.cuda.empty_cache()
                timing["gpu_service_seconds"] += time.monotonic() - started


def checkpoints(plan):
    root = Path(plan["output_dir"])
    result = {f"rl-{number}": root / f"round-{number}/checkpoint"
              for number in range(1, 6)}
    required = ("adapter_model.safetensors", "training-state.pt", "training_manifest.json")
    for label, path in result.items():
        if not all((path / name).is_file() for name in required):
            raise FileNotFoundError(f"incomplete checkpoint for {label}: {path}")
    return result


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-concurrency", type=int, default=3)
    parser.add_argument("--expert-concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    if args.gpu_concurrency < 1 or args.expert_concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint evaluation requires CUDA")
    plan = json.loads(Path(args.plan).read_text())
    tasks = load_tasks(args.tasks)
    validate_tasks(tasks)
    model_paths = checkpoints(plan)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=args.resume)
    os.environ["TRADINGAGENTS_OHLCV_SNAPSHOT_DIR"] = str(Path(args.snapshot_dir).resolve())
    manifest = {
        "models": {name: str(path) for name, path in model_paths.items()},
        "tasks": tasks, "planned_executions": len(model_paths) * len(tasks),
        "comparison": "exact SFT seven-task benchmark across RL-1 through RL-5",
        "modes": ["learned"], "gpu_concurrency": args.gpu_concurrency,
        "expert_concurrency": args.expert_concurrency,
        "sft_baseline_summary": "artifacts/scheduler/sft-benchmark-21-20260905/summary.json",
        "exposure_note": "seen_sft/unseen_sft labels describe SFT exposure only; some tasks were later used by RL.",
    }
    manifest_path = output / "manifest.json"
    if args.resume:
        existing = json.loads(manifest_path.read_text())
        for field in ("models", "tasks", "gpu_concurrency", "expert_concurrency"):
            if existing[field] != manifest[field]:
                raise ValueError(f"resume configuration changed: {field}")
    write_json_atomic(manifest_path, manifest)
    torch.set_num_threads(2)
    slots = BoundedSemaphore(args.gpu_concurrency)
    policies = {}
    for label, checkpoint in model_paths.items():
        model, tokenizer, _ = load_scheduler_model(SchedulerModelConfig(
            base_model=plan["base_model"], base_revision=None,
            adapter_path=str(checkpoint)), training=False)
        model.config.use_cache = False
        model.requires_grad_(False)
        policies[label] = ConcurrentCheckpointPolicy(
            model, tokenizer, policy_id=f"unseen-eval:{label}", device="cpu",
            gpu_slots=slots, max_context_tokens=32768,
        )
    runtime = scheduler_runtime_config({
        "llm_provider": "openrouter", "quick_think_llm": "z-ai/glm-5.3-flash",
        "deep_think_llm": "z-ai/glm-5.3-flash", "temperature": 0.0,
        "llm_max_retries": 1, "scheduler_max_steps": 16,
        "scheduler_max_context_tokens": 32768,
        "max_debate_rounds": 1, "max_risk_discuss_rounds": 1,
    })
    values, retrying = (prepare_resume(output, policies, tasks)
                        if args.resume else ({label: [] for label in policies}, []))
    completed = {(label, item.task_id) for label, items in values.items() for item in items}
    write_json_atomic(output / "status.json", {
        "status": "running", "finished": len(completed),
        "planned": manifest["planned_executions"],
    })
    with limit_expert_requests(args.expert_concurrency), ThreadPoolExecutor(
            max_workers=manifest["planned_executions"]) as pool:
        futures = {
            pool.submit(run_one, task, label, policy, runtime, output): label
            for label, policy in policies.items() for task in tasks
            if (label, task["task_id"]) not in completed
        }
        for future in as_completed(futures):
            values[futures[future]].append(future.result())
            write_json_atomic(output / "status.json", {
                "status": "running", "finished": sum(map(len, values.values())),
                "planned": manifest["planned_executions"],
            })
    reviewer = AutomaticReviewer()
    reviews = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(reviewer.assess, trajectory): trajectory
                   for items in values.values() for trajectory in items}
        for future in as_completed(futures):
            trajectory = futures[future]
            reviews[trajectory.trajectory_id] = future.result()
            write_json_atomic(output / "quality-reviews.json", reviews)
    write_json_atomic(output / "summary.json", summarize(values, tasks, reviews))
    write_json_atomic(output / "status.json", {
        "status": "completed", "finished": manifest["planned_executions"],
        "planned": manifest["planned_executions"],
        "successful": sum(item.execution_status == "completed"
                          for items in values.values() for item in items),
    })


if __name__ == "__main__":
    main()
