"""Three sequential on-policy RL rounds, each followed by two fixed evaluations."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv

from tradingagents.scheduler.store import write_json_atomic
from .collect_rollouts import RolloutConfig
from .generate import load_tasks
from .model import SchedulerModelConfig, load_scheduler_model
from .train_grpo import GRPOTrainConfig
from .train_with_evaluation import evaluate_epoch


def validate_plan(plan):
    training = load_tasks(plan["train_tasks"], split="train")
    evaluation = load_tasks(plan["eval_tasks"])
    train_ids = {task["task_id"] for task in training}
    eval_ids = {task["task_id"] for task in evaluation}
    if len(training) != 8 or len(train_ids) != 8 or len(evaluation) != 2 or len(eval_ids) != 2:
        raise ValueError("RL requires eight distinct training tasks and two distinct evaluation tasks")
    if train_ids & eval_ids:
        raise ValueError("RL training/evaluation task leakage")
    return evaluation


def run_stage(module, config, directory):
    path = directory / f"{module.rsplit('.', 1)[-1]}.json"
    write_json_atomic(path, asdict(config))
    with (path.with_suffix(".log")).open("w") as log:
        subprocess.run([sys.executable, "-m", module, "--config", str(path)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def evaluate_checkpoint(plan, adapter, tasks, number, directory):
    import torch

    model, tokenizer, _ = load_scheduler_model(SchedulerModelConfig(
        base_model=plan["base_model"], base_revision=None, adapter_path=str(adapter),
    ), training=False)
    model.to("cuda")
    model.config.use_cache = False
    evaluate_epoch(number, model, tokenizer, Path(adapter), tasks=tasks,
                   output=directory, max_length=32768,
                   include_static=False, include_teacher=False)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return json.loads((directory / f"epoch-{number:02d}" / "comparison.json").read_text())


def run(plan):
    import torch

    tasks = validate_plan(plan)
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available; RL was not started")
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "plan.json", plan)
    os.environ["TRADINGAGENTS_OHLCV_SNAPSHOT_DIR"] = str(Path(plan["snapshot_dir"]).resolve())
    initial = plan["initial_adapter"]
    active = initial
    number = 0
    try:
        write_json_atomic(output / "status.json", {"status": "running", "round": 0, "stage": "sft_baseline"})
        baseline = evaluate_checkpoint(plan, initial, tasks, 0, output / "sft-baseline")
        if baseline["modes"]["learned"]["completion_rate"] == 0:
            raise RuntimeError("Both SFT baseline tasks failed; repair task execution before RL")
        for number in range(1, 4):
            directory = output / f"round-{number}"
            directory.mkdir()
            write_json_atomic(output / "status.json", {"status": "running", "round": number, "stage": "rollout"})
            rollout = RolloutConfig(
                tasks_path=plan["train_tasks"], static_trajectories_path="",
                active_adapter_path=active, reference_adapter_path=initial,
                output_dir=str(directory / "rollouts"), run_id=f"rl-round-{number}",
                base_model=plan["base_model"], base_revision=None,
                reward_mode="automatic", reward_config_path=None, group_size=4,
                seed=42 + number * 100, action_temperature=0.8,
            )
            run_stage("training.scheduler.collect_rollouts", rollout, directory)
            counts = json.loads((directory / "rollouts/rollout_manifest.json").read_text())["counts"]
            if counts["trajectories"] < 8 or counts["rows"] == 0:
                raise RuntimeError("Fewer than two usable reward groups; no parameter update performed")
            checkpoint = directory / "checkpoint"
            training = GRPOTrainConfig(
                rollout_path=str(directory / "rollouts/grpo.jsonl"), sft_adapter_path=active,
                output_dir=str(checkpoint), base_model=plan["base_model"], base_revision=None,
                learning_rate=5e-6, epochs=1, group_size=4,
            )
            write_json_atomic(output / "status.json", {"status": "running", "round": number, "stage": "grpo_update"})
            run_stage("training.scheduler.train_grpo", training, directory)
            active = str(checkpoint)
            write_json_atomic(output / "status.json", {"status": "running", "round": number, "stage": "evaluation"})
            evaluate_checkpoint(plan, active, tasks, number, directory / "evaluation")
        write_json_atomic(output / "status.json", {"status": "completed", "rounds": 3, "final_adapter": active})
    except BaseException as exc:
        write_json_atomic(output / "status.json", {"status": "needs_attention", "round": number,
                          "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    load_dotenv()
    run(json.loads(Path(args.plan).read_text()))


if __name__ == "__main__":
    main()
