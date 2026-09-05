"""Frozen baseline/third/final SFT policies on seven real tasks, exactly once."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
from threading import BoundedSemaphore, Lock
import time

from dotenv import load_dotenv

from tradingagents.scheduler.hf_policy import HFSchedulerPolicy
from tradingagents.scheduler.store import write_json_atomic
from .auto_review import AutomaticReviewer
from .evaluate import evaluate_trajectories
from .generate import load_tasks
from .gpu_lease import gpu_lease
from .model import SchedulerModelConfig, load_scheduler_model
from .runtime_config import scheduler_runtime_config
from .train_with_evaluation import run_task


@contextmanager
def limit_expert_requests(limit=8):
    from langchain_openai import ChatOpenAI

    semaphore = BoundedSemaphore(limit)
    original = ChatOpenAI._generate

    def limited(self, *args, **kwargs):
        with semaphore:
            return original(self, *args, **kwargs)

    ChatOpenAI._generate = limited
    try:
        yield
    finally:
        ChatOpenAI._generate = original


class BenchmarkPolicy(HFSchedulerPolicy):
    def __init__(self, *args, gpu_lock, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_lock = gpu_lock
        self.timings = defaultdict(lambda: {"queue_seconds": 0.0, "gpu_service_seconds": 0.0})

    def _distribution(self, context):
        import torch

        requested = time.monotonic()
        with self.gpu_lock, gpu_lease():
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


def validate_tasks(tasks):
    ids = {task["task_id"] for task in tasks}
    buckets = [task.get("evaluation_bucket") for task in tasks]
    if len(tasks) != 7 or len(ids) != 7 or buckets.count("seen_sft") != 3 or buckets.count("unseen_sft") != 4:
        raise ValueError("benchmark requires exactly three seen and four unseen tasks")
    for task in tasks:
        counts = (task["sft_part3_samples_seen"], task["sft_part6_samples_seen"])
        if task["evaluation_bucket"] == "seen_sft" and not all(count > 0 for count in counts):
            raise ValueError("seen task must be present in both SFT checkpoints")
        if task["evaluation_bucket"] == "unseen_sft" and any(counts):
            raise ValueError("unseen task overlaps SFT training")


def wait_for_training(training_status, output):
    """No GPU allocation while the independently running RL pipeline owns it."""
    import torch

    while True:
        state = json.loads(training_status.read_text()) if training_status.exists() else {}
        running = state.get("status") in (None, "running")
        if not running and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if free >= 18 * 1024**3:
                return
        write_json_atomic(output / "status.json", {
            "status": "waiting_for_gpu", "planned_tasks": 21,
            "training_stage": state.get("stage"), "training_round": state.get("round"),
        })
        time.sleep(15)


def run_one(task, label, policy, runtime, output):
    started = time.monotonic()
    # Keep exposure/stratification metadata out of the scheduler task input.
    fields = ("task_id", "ticker", "trade_date", "asset_type", "selected_analysts",
              "research_depth", "output_language", "data_snapshot_id", "information_cutoff", "split")
    initial = {key: task[key] for key in fields if key in task}
    trajectory = run_task(initial, "learned", policy, runtime, output / label, f"benchmark-{label}")
    timing = {"wall_seconds": time.monotonic() - started,
              **policy.timings.get(task["task_id"], {}),
              "gpu_service_includes_weight_transfers": True}
    write_json_atomic(output / label / task["task_id"] / "timing.json", timing)
    return trajectory


def summarize(values, tasks, reviews):
    buckets = {task["task_id"]: task["evaluation_bucket"] for task in tasks}
    summary = {}
    for label, trajectories in values.items():
        summary[label] = {}
        for bucket in ("seen_sft", "unseen_sft"):
            selected = [t for t in trajectories if buckets[t.task_id] == bucket]
            metrics = evaluate_trajectories(selected, {})
            for key in ("trader_match_rate", "mean_portfolio_rating_distance", "same_as_static_path_rate"):
                metrics.pop(key, None)
            scores = [reviews[t.trajectory_id]["quality"] for t in selected
                      if reviews[t.trajectory_id]["status"] == "reviewed"]
            metrics.update(quality_reviewed=len(scores), mean_quality=sum(scores)/len(scores) if scores else None,
                           api_cost_usd=None, cost_note="Token counts recorded; dollar cost not inferred from text length.")
            summary[label][bucket] = metrics
    return summary


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--training-status", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--share-gpu", action="store_true")
    parser.add_argument("--gpu-budget-gib", type=float, default=7.0)
    args = parser.parse_args()
    load_dotenv()
    tasks = load_tasks(args.tasks)
    validate_tasks(tasks)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "manifest.json", {"tasks": tasks, "model_versions": ["base", "sft3", "sft6"],
        "seed": 42, "planned_tasks": 21, "expert_request_limit": 8, "gpu_concurrency": 1,
        "raw_base_action_tokens": "same project action tokens registered; model loading/resizing seed 42; no SFT adapter",
        "shared_gpu": args.share_gpu, "gpu_allocator_budget_gib": args.gpu_budget_gib,
        "checkpoint_root": args.checkpoint_root, "base_model": args.base_model})
    os.environ["TRADINGAGENTS_OHLCV_SNAPSHOT_DIR"] = str(Path(args.snapshot_dir).resolve())
    if args.share_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU evaluation requires CUDA")
        total = torch.cuda.get_device_properties(0).total_memory
        fraction = args.gpu_budget_gib * 1024**3 / total
        if not 0 < fraction < 1:
            raise ValueError("GPU budget must be smaller than total device memory")
        torch.cuda.set_per_process_memory_fraction(fraction)
    else:
        wait_for_training(Path(args.training_status), output)
    torch.set_num_threads(2)
    gpu_lock = Lock()
    policies = {}
    for label, part in (("base", None), ("sft3", 3), ("sft6", 6)):
        torch.manual_seed(42)
        adapter = str(Path(args.checkpoint_root) / f"checkpoint-epoch-1-part-{part}") if part else None
        model, tokenizer, _ = load_scheduler_model(SchedulerModelConfig(
            base_model=args.base_model, base_revision=None, adapter_path=adapter), training=False)
        model.config.use_cache = False
        model.requires_grad_(False)
        policies[label] = BenchmarkPolicy(model, tokenizer, policy_id=f"sft-benchmark:{label}",
                                          device="cpu", gpu_lock=gpu_lock, max_context_tokens=32768)
    runtime = scheduler_runtime_config({"llm_provider": "openrouter", "quick_think_llm": "z-ai/glm-5.3-flash",
        "deep_think_llm": "z-ai/glm-5.3-flash", "temperature": 0.0, "llm_max_retries": 1,
        "scheduler_max_steps": 16, "scheduler_max_context_tokens": 32768,
        "max_debate_rounds": 1, "max_risk_discuss_rounds": 1})
    values = {label: [] for label in policies}
    write_json_atomic(output / "status.json", {"status": "running", "finished": 0, "planned": 21})
    with limit_expert_requests(8), ThreadPoolExecutor(max_workers=21) as pool:
        futures = {pool.submit(run_one, task, label, policy, runtime, output): label
                   for label, policy in policies.items() for task in tasks}
        for future in as_completed(futures):
            values[futures[future]].append(future.result())
            write_json_atomic(output / "status.json", {"status": "running", "finished": sum(map(len, values.values())), "planned": 21})
    reviewer = AutomaticReviewer()
    reviews = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(reviewer.assess, t): t for ts in values.values() for t in ts}
        for future in as_completed(futures):
            reviews[futures[future].trajectory_id] = future.result()
            write_json_atomic(output / "quality-reviews.json", reviews)
    write_json_atomic(output / "summary.json", summarize(values, tasks, reviews))
    write_json_atomic(output / "status.json", {"status": "completed", "finished": 21, "planned": 21,
        "successful_tasks": sum(t.execution_status == "completed" for ts in values.values() for t in ts)})


if __name__ == "__main__":
    main()
