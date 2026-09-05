"""Re-evaluate saved checkpoints with frozen OHLCV, without restarting SFT."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv

from tradingagents.scheduler.store import write_json_atomic
from .generate import load_tasks
from .model import SchedulerModelConfig, load_scheduler_model, scheduler_input_ids
from .train_sft import SFTTrainConfig
from .train_with_evaluation import evaluate_epoch, validate_eval_tasks


def select_inference_device(policy, context, output):
    """Keep training GPU untouched; long CPU contexts wait for training to finish."""
    import torch

    if policy.device == "cuda":
        return
    length = len(scheduler_input_ids(policy.tokenizer, context.serialized_state))
    if length > policy.max_context_tokens:
        return  # The policy reports the normal context-overflow error.
    announced = False
    while True:
        status = json.loads((output / "training-status.json").read_text())["status"]
        if status in {"completed", "failed"} and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if free >= 12 * 1024**3:
                policy.model.to("cuda")
                policy.device = "cuda"
                write_json_atomic(output / "evaluation-repair-device.json",
                                  {"device": "cuda", "waiting_for_training": False})
                return
        if length <= 8192:
            return
        if not announced:
            write_json_atomic(output / "evaluation-repair-device.json",
                              {"device": "cpu", "waiting_for_training": True, "tokens": length})
            print(json.dumps({"event": "long_context_waiting_for_training_gpu", "tokens": length}), flush=True)
            announced = True
        time.sleep(5)


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--learned-only", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    os.environ["TRADINGAGENTS_OHLCV_SNAPSHOT_DIR"] = str(Path(args.snapshot_dir).resolve())
    torch.set_num_threads(4)
    config = SFTTrainConfig.from_json(args.config)
    tasks = load_tasks(args.tasks)
    validate_eval_tasks(tasks, config.train_path, config.validation_path)
    output = Path(config.output_dir)
    status_file = output / ("evaluation-learned-status.json" if args.learned_only else "evaluation-repair-status.json")
    scenario_dir = "scenarios-learned-only" if args.learned_only else "scenarios-retry-1"
    write_json_atomic(status_file, {"status": "running", "finished_checkpoints": 0, "pid": os.getpid()})
    try:
        for part in range(1, 7):
            request = output / "evaluation-queue" / f"part-{part}.json"
            while not request.exists():
                training = json.loads((output / "training-status.json").read_text())
                if training["status"] != "running":
                    raise RuntimeError(f"training ended without checkpoint {part}")
                time.sleep(2)
            checkpoint = Path(json.loads(request.read_text())["checkpoint"])
            model, tokenizer, _ = load_scheduler_model(SchedulerModelConfig(
                base_model=config.base_model, base_revision=config.base_revision,
                adapter_path=str(checkpoint), dtype=config.dtype,
                attention_implementation=config.attention_implementation,
            ), training=False)
            model.config.use_cache = False
            evaluate_epoch(
                1, model, tokenizer, checkpoint, tasks=tasks, output=output / scenario_dir,
                max_length=config.max_length, include_teacher=not args.learned_only,
                include_static=not args.learned_only, stage_name=f"part-{part}",
                before_select=lambda policy, context: select_inference_device(policy, context, output),
            )
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            write_json_atomic(status_file, {"status": "running", "finished_checkpoints": part, "pid": os.getpid()})
        write_json_atomic(status_file, {"status": "completed", "finished_checkpoints": 6,
                                       "attempts": 30 if args.learned_only else 90})
    except BaseException as exc:
        write_json_atomic(status_file, {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
