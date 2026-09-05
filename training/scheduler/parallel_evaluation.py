"""Run all six checkpoint task groups concurrently, sharing one GPU safely."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json
import os
from pathlib import Path
import shutil
from threading import Lock

from dotenv import load_dotenv

from tradingagents.scheduler.hf_policy import HFSchedulerPolicy
from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from .generate import load_tasks
from .model import SchedulerModelConfig, load_scheduler_model
from .train_sft import SFTTrainConfig
from .train_with_evaluation import evaluate_epoch, validate_eval_tasks

MODEL_LOAD_LOCK = Lock()
MODEL_CACHE = {}


class SharedGPUModelPolicy(HFSchedulerPolicy):
    """Keep checkpoint weights in RAM; only one forward pass occupies GPU memory."""

    def __init__(self, *args, gpu_lock, **kwargs):
        with gpu_lock:
            super().__init__(*args, **kwargs)
        self.gpu_lock = gpu_lock

    def _distribution(self, context):
        import torch

        with self.gpu_lock:
            try:
                self.model.to("cuda")
                self.device = "cuda"
                return super()._distribution(context).cpu()
            finally:
                self.model.to("cpu")
                self.device = "cpu"
                torch.cuda.empty_cache()


def run_part(part, config, tasks, gpu_lock, selected_only=False):
    output = Path(config.output_dir)
    checkpoint = output / f"checkpoint-epoch-1-part-{part}"
    # Transformers/PEFT lazy imports and initialization mutate process-global state.
    # Serialize initialization only; task execution remains concurrent.
    with MODEL_LOAD_LOCK:
        if str(checkpoint) not in MODEL_CACHE:
            MODEL_CACHE[str(checkpoint)] = load_scheduler_model(SchedulerModelConfig(
                base_model=config.base_model, base_revision=config.base_revision,
                adapter_path=str(checkpoint), dtype=config.dtype,
                attention_implementation=config.attention_implementation,
            ), training=False)
        model, tokenizer, _ = MODEL_CACHE[str(checkpoint)]
    model.config.use_cache = False
    prior = "scenarios-learned-parallel-paused" if (output / "scenarios-learned-parallel-paused").exists() else "scenarios-learned-only"
    previous = output / ("no-previous-extra-attempt" if selected_only else prior) / f"part-{part}"
    saved = []
    for path in previous.glob("*/learned/raw.jsonl"):
        saved.extend(t for t in TrajectoryStore(path).load()
                     if not selected_only or t.execution_status == "completed")
    destination = output / ("scenarios-learned-parts-3-6" if selected_only else "scenarios-learned-parallel")
    evaluate_epoch(
        1, model, tokenizer, checkpoint, tasks=tasks, output=destination,
        stage_name=f"part-{part}", max_length=config.max_length,
        include_static=False, include_teacher=False,
        policy_factory=partial(SharedGPUModelPolicy, gpu_lock=gpu_lock),
        existing_results=saved,
    )
    for path in previous.glob("*/learned/raw.jsonl"):
        if path.parent.parent.name not in {t.task_id for t in saved}:
            continue
        target = destination / f"part-{part}" / path.parent.parent.name / "learned"
        shutil.copytree(path.parent, target, dirs_exist_ok=False)
    report_path = destination / f"part-{part}" / "comparison.json"
    report = json.loads(report_path.read_text())
    report.update(learned_device="cuda", weight_storage="cpu", gpu_forward_concurrency=1,
                  interpretation="Learned-only task evaluation; no Static or Teacher comparison. GPU forwards are serialized across concurrent task groups.")
    write_json_atomic(report_path, report)
    return part


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--parts-three-six", action="store_true")
    parser.add_argument("--extra-three-six", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    os.environ["TRADINGAGENTS_OHLCV_SNAPSHOT_DIR"] = str(Path(args.snapshot_dir).resolve())
    torch.set_num_threads(2)
    config = SFTTrainConfig.from_json(args.config)
    tasks = load_tasks(args.tasks)
    validate_eval_tasks(tasks, config.train_path, config.validation_path)
    output = Path(config.output_dir)
    if json.loads((output / "training-status.json").read_text())["status"] != "completed":
        raise RuntimeError("parallel GPU evaluation requires SFT to have completed")
    if not torch.cuda.is_available():
        raise RuntimeError("parallel evaluation requires the server GPU")
    parts = [3, 6] if args.parts_three_six else list(range(1, 7))
    jobs = [(part, args.parts_three_six) for part in parts]
    if args.extra_three_six:
        if args.parts_three_six:
            raise ValueError("extra and selected-only modes are mutually exclusive")
        jobs += [(3, True), (6, True)]
    status = output / ("evaluation-parts-3-6-status.json" if args.parts_three_six else "evaluation-parallel-status.json")
    if status.exists():
        raise RuntimeError("parallel evaluation already exists; refusing duplicate launch")
    completed, errors = [], []
    gpu_lock = Lock()
    write_json_atomic(status, {"status": "running", "pid": os.getpid(),
                              "checkpoint_workers": len(jobs), "task_workers": len(jobs) * len(tasks)})
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(run_part, part, config, tasks, gpu_lock, extra): (part, extra) for part, extra in jobs}
        for future in as_completed(futures):
            part, extra = futures[future]
            try:
                future.result()
                completed.append(f"extra-{part}" if args.extra_three_six and extra else str(part))
            except Exception as exc:
                errors.append({"part": part, "error_type": type(exc).__name__, "error": str(exc)})
            write_json_atomic(status, {"status": "running", "pid": os.getpid(),
                "completed_parts": sorted(completed), "errors": errors,
                "checkpoint_workers": len(jobs), "task_workers": len(jobs) * len(tasks)})
    write_json_atomic(status, {"status": "failed" if errors else "completed",
        "completed_parts": sorted(completed), "errors": errors, "planned_tasks": len(jobs) * len(tasks)})
    if errors:
        raise RuntimeError("some checkpoint evaluations failed; see parallel status")


if __name__ == "__main__":
    main()
