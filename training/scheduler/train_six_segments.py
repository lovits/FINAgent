"""One SFT epoch with six immutable checkpoints and a concurrent CPU evaluator."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dotenv import load_dotenv

from tradingagents.scheduler.store import write_json_atomic
from .generate import load_tasks
from .model import SchedulerModelConfig, load_scheduler_model
from .train_sft import SFTTrainConfig, train
from .train_with_evaluation import evaluate_epoch, validate_eval_tasks

SEGMENTS = 6


def publish_checkpoint(output: Path, segment: int, checkpoint: Path) -> None:
    """Publish only after adapter, tokenizer and optimizer state have been saved."""
    position = json.loads((checkpoint / "training-position.json").read_text())
    write_json_atomic(
        output / "evaluation-queue" / f"part-{segment}.json",
        {"segment": segment, "checkpoint": str(checkpoint.resolve()),
         "covered_samples": position["covered_samples"]},
    )
    print(json.dumps({"event": "evaluation_queued", "segment": segment,
                      "covered_samples": position["covered_samples"]}), flush=True)


def evaluate_queue(config, tasks) -> None:
    import torch

    torch.set_num_threads(4)
    output = Path(config.output_dir)
    errors = []
    for segment in range(1, SEGMENTS + 1):
        request = output / "evaluation-queue" / f"part-{segment}.json"
        while not request.exists():
            status = json.loads((output / "training-status.json").read_text())
            if status["status"] != "running":
                raise RuntimeError(f"training ended without checkpoint {segment}")
            time.sleep(2)
        checkpoint = Path(json.loads(request.read_text())["checkpoint"])
        stage = f"part-{segment}"
        model = tokenizer = None
        try:
            model, tokenizer, _ = load_scheduler_model(
                SchedulerModelConfig(
                    base_model=config.base_model, base_revision=config.base_revision,
                    adapter_path=str(checkpoint), dtype=config.dtype,
                    attention_implementation=config.attention_implementation,
                ),
                training=False,
            )
            model.config.use_cache = False
            evaluate_epoch(
                1, model, tokenizer, checkpoint, tasks=tasks,
                output=output / "scenarios", max_length=config.max_length,
                include_teacher=True, stage_name=stage,
            )
        except Exception as exc:
            failure = {"segment": segment, "error_type": type(exc).__name__,
                       "error": str(exc)}
            errors.append(failure)
            write_json_atomic(output / "scenarios" / stage / "evaluation-error.json", failure)
            print(json.dumps({"event": "evaluation_error", **failure}), flush=True)
        finally:
            del model, tokenizer
            gc.collect()
    if errors:
        raise RuntimeError(f"{len(errors)} checkpoint evaluations failed; see evaluation-error.json")


def run(config, config_path: str, tasks_path: str) -> None:
    output = Path(config.output_dir)
    if config.epochs != 1 or not config.cover_all_samples:
        raise ValueError("six-segment run requires exactly one full-coverage epoch")
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "training-status.json", {"status": "running"})
    environment = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
    }
    with (output / "evaluation.log").open("w") as log:
        worker = subprocess.Popen(
            [sys.executable, "-m", "training.scheduler.train_six_segments", "--config", config_path,
             "--tasks", tasks_path, "--worker"],
            env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        write_json_atomic(output / "worker.json", {"pid": worker.pid, "device": "cpu"})
        try:
            metrics = train(
                config, segments=SEGMENTS,
                segment_callback=lambda part, checkpoint: publish_checkpoint(output, part, checkpoint),
            )
        except BaseException as exc:
            write_json_atomic(output / "training-status.json",
                              {"status": "failed", "error_type": type(exc).__name__})
            raise
        write_json_atomic(output / "training-status.json", {"status": "completed", "metrics": metrics})
        print(json.dumps({"event": "training_completed", "metrics": metrics,
                          "evaluation_still_running": worker.poll() is None}), flush=True)
        if worker.wait() != 0:
            raise RuntimeError("CPU evaluation worker failed; training checkpoints are preserved")
    write_json_atomic(output / "cycle-complete.json",
                      {"status": "completed", "checkpoints": SEGMENTS,
                       "evaluation_tasks": SEGMENTS * 5 * 3, "metrics": metrics})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    config = SFTTrainConfig.from_json(args.config)
    tasks = load_tasks(args.tasks)
    validate_eval_tasks(tasks, config.train_path, config.validation_path)
    if args.worker:
        evaluate_queue(config, tasks)
    else:
        run(config, args.config, args.tasks)


if __name__ == "__main__":
    main()
