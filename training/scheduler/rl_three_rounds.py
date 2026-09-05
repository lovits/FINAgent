"""Sequential on-policy RL training rounds with independent checkpoints."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv

from tradingagents.scheduler.store import write_json_atomic
from .collect_rollouts import RolloutConfig
from .generate import load_tasks
from .train_grpo import GRPOTrainConfig


def validate_plan(plan):
    training = load_tasks(plan["train_tasks"], split="train")
    train_ids = {task["task_id"] for task in training}
    if len(training) != 8 or len(train_ids) != 8:
        raise ValueError("RL requires eight distinct training tasks")
    return training


def run_stage(module, config, directory):
    path = directory / f"{module.rsplit('.', 1)[-1]}.json"
    write_json_atomic(path, asdict(config))
    with (path.with_suffix(".log")).open("w") as log:
        subprocess.run([sys.executable, "-m", module, "--config", str(path)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def _completed_checkpoint(output: Path, number: int) -> Path | None:
    checkpoint = output / f"round-{number}" / "checkpoint"
    required = ("adapter_model.safetensors", "training-state.pt", "training_manifest.json")
    return checkpoint if all((checkpoint / name).is_file() for name in required) else None


def run(plan, *, resume=False):
    import torch

    validate_plan(plan)
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available; RL was not started")
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=resume)
    if not resume or not (output / "plan.json").exists():
        write_json_atomic(output / "plan.json", plan)
    os.environ["TRADINGAGENTS_OHLCV_SNAPSHOT_DIR"] = str(Path(plan["snapshot_dir"]).resolve())
    initial = plan["initial_adapter"]
    total_rounds = int(plan.get("rounds", 5))
    if total_rounds < 1:
        raise ValueError("rounds must be positive")
    active = initial
    start_round = 1
    for completed_round in range(1, total_rounds + 1):
        checkpoint = _completed_checkpoint(output, completed_round)
        if checkpoint is None:
            break
        active = str(checkpoint)
        start_round = completed_round + 1
    if start_round > total_rounds:
        raise ValueError("all GRPO rounds are already complete")
    number = start_round
    try:
        for number in range(start_round, total_rounds + 1):
            directory = output / f"round-{number}"
            directory.mkdir(exist_ok=resume and number == start_round)
            write_json_atomic(output / "status.json", {"status": "running", "round": number, "stage": "rollout"})
            rollout = RolloutConfig(
                tasks_path=plan["train_tasks"], static_trajectories_path="",
                active_adapter_path=active, reference_adapter_path=initial,
                output_dir=str(directory / "rollouts"), run_id=f"rl-round-{number}",
                base_model=plan["base_model"], base_revision=None,
                reward_mode="automatic", reward_config_path=None, group_size=4,
                seed=42 + number * 100, action_temperature=0.8,
                parallel_workers=8, trajectory_workers=4,
                resume=resume and number == start_round,
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
            write_json_atomic(output / "status.json", {
                "status": "running", "round": number,
                "stage": "checkpoint_saved", "checkpoint": active,
            })
        write_json_atomic(output / "status.json", {
            "status": "completed", "rounds": total_rounds, "final_adapter": active,
        })
    except BaseException as exc:
        write_json_atomic(output / "status.json", {"status": "needs_attention", "round": number,
                          "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    run(json.loads(Path(args.plan).read_text()), resume=args.resume)


if __name__ == "__main__":
    main()
