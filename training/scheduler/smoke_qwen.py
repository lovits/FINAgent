"""Progressive 2K-to-32K Qwen3 scheduler backward smoke for one RTX 4090."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from time import monotonic

from tradingagents.scheduler.store import write_json_atomic

from .model import SchedulerModelConfig, load_scheduler_model
from .sft_loss import masked_action_cross_entropy


@dataclass(frozen=True)
class QwenSmokeConfig:
    model_path: str = "/root/autodl-tmp/models/Qwen3-1.7B"
    output_path: str = "artifacts/scheduler/qwen3-1p7b/smoke/memory_report.json"
    lengths: tuple[int, ...] = (2048, 8192, 16384, 32768)
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"


def run_smoke(config: QwenSmokeConfig) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen 32K smoke requires a CUDA GPU")
    model, _, action_ids = load_scheduler_model(
        SchedulerModelConfig(
            base_model=config.model_path,
            base_revision=None,
            tokenizer_revision=None,
            dtype=config.dtype,
            attention_implementation=config.attention_implementation,
        ),
        training=True,
    )
    model.to("cuda")
    model.train()
    results = []
    for length in config.lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.zero_grad(set_to_none=True)
        input_ids = torch.full((1, length), 2, dtype=torch.long, device="cuda")
        attention_mask = torch.ones_like(input_ids)
        valid_ids = torch.tensor([action_ids[:2]], device="cuda")
        started = monotonic()
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, _ = masked_action_cross_entropy(
                outputs.logits,
                torch.tensor([length - 1], device="cuda"),
                valid_ids,
                torch.ones_like(valid_ids, dtype=torch.bool),
                torch.tensor([action_ids[1]], device="cuda"),
            )
            loss.backward()
            torch.cuda.synchronize()
            results.append(
                {
                    "length": length,
                    "status": "passed",
                    "loss": float(loss.detach()),
                    "elapsed_seconds": monotonic() - started,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                }
            )
        except torch.OutOfMemoryError:
            results.append(
                {
                    "length": length,
                    "status": "oom",
                    "elapsed_seconds": monotonic() - started,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            break
    report = {
        "config": asdict(config),
        "device": torch.cuda.get_device_name(0),
        "results": results,
        "all_passed": len(results) == len(config.lengths)
        and all(value["status"] == "passed" for value in results),
    }
    write_json_atomic(config.output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/models/Qwen3-1.7B",
    )
    parser.add_argument(
        "--output",
        default="artifacts/scheduler/qwen3-1p7b/smoke/memory_report.json",
    )
    parser.add_argument("--lengths", default="2048,8192,16384,32768")
    parser.add_argument("--attention-implementation", default="sdpa")
    args = parser.parse_args()
    config = QwenSmokeConfig(
        model_path=args.model_path,
        output_path=args.output,
        lengths=tuple(int(value) for value in args.lengths.split(",")),
        attention_implementation=args.attention_implementation,
    )
    report = run_smoke(config)
    print(json.dumps(report, sort_keys=True))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
