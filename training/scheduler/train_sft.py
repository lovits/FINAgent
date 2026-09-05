"""Single-GPU masked-action LoRA SFT for the scheduler policy."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

from tradingagents.scheduler.store import write_json_atomic

from .model import SchedulerModelConfig, load_scheduler_model
from .sft_dataset import (
    HierarchicalSourceSampler,
    MaskedActionCollator,
    SchedulerSFTDataset,
    validate_task_isolation,
)
from .sft_loss import masked_action_cross_entropy


@dataclass(frozen=True)
class SFTTrainConfig:
    train_path: str
    validation_path: str
    output_dir: str
    base_model: str = "Qwen/Qwen3-1.7B"
    base_revision: str | None = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    tokenizer_revision: str | None = None
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    max_length: int = 32768
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    epochs: int = 2
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    teacher_probability: float = 0.8
    cover_all_samples: bool = True
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> SFTTrainConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def checkpoint_boundaries(batches: int, accumulation: int, segments: int) -> list[int]:
    if min(batches, accumulation, segments) < 1:
        raise ValueError("checkpoint dimensions must be positive")
    boundaries = [
        min(batches, math.ceil(batches * part / (segments * accumulation)) * accumulation)
        for part in range(1, segments + 1)
    ]
    if len(set(boundaries)) != segments:
        raise ValueError("not enough optimizer updates for distinct checkpoint segments")
    return boundaries


def train(config: SFTTrainConfig, *, epoch_callback=None, half_epoch_callback=None,
          segments: int = 0, segment_callback=None) -> dict[str, float]:
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_data = SchedulerSFTDataset(config.train_path)
    validation_data = SchedulerSFTDataset(config.validation_path)
    validate_task_isolation(train_data.rows, validation_data.rows)
    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
    if config.cover_all_samples and accelerator.num_processes != 1:
        raise ValueError("full-coverage SFT currently supports one training process")
    model, tokenizer, _ = load_scheduler_model(
        SchedulerModelConfig(
            base_model=config.base_model,
            base_revision=config.base_revision,
            tokenizer_revision=config.tokenizer_revision,
            dtype=config.dtype,
            attention_implementation=config.attention_implementation,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        ),
        training=True,
    )
    sampler = HierarchicalSourceSampler(
        train_data.rows,
        teacher_probability=config.teacher_probability,
        seed=config.seed,
        cover_all=config.cover_all_samples,
    )
    collator = MaskedActionCollator(tokenizer, max_length=config.max_length)

    def tracked_collator(rows):
        return {**collator(rows), "sample_ids": [row["sample_id"] for row in rows]}

    train_loader = DataLoader(
        train_data,
        batch_size=config.micro_batch_size,
        sampler=sampler,
        collate_fn=tracked_collator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.micro_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    boundaries = (
        checkpoint_boundaries(len(train_loader), config.gradient_accumulation_steps, segments)
        if segments else []
    )
    total_updates = max(1, updates_per_epoch * config.epochs)
    source_by_id = {
        row["sample_id"]: "static" if row["source"] == "static" else "teacher"
        for row in train_data.rows
    }
    if len(source_by_id) != len(train_data):
        raise ValueError("SFT sample_id values must be unique")
    if accelerator.is_main_process:
        write_json_atomic(
            Path(config.output_dir) / "sampling_plan.json",
            {
                "unique_samples": len(train_data),
                "cover_all_samples": config.cover_all_samples,
                "sampling_method": "shuffle_without_replacement" if config.cover_all_samples
                else "legacy_source_weighted",
                "source_draws_per_epoch": sampler.source_quotas(),
                "draws_per_epoch": len(sampler),
                "planned_updates": total_updates,
                "epochs": config.epochs,
                "checkpoint_batches": boundaries,
            },
        )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        validation_loader,
        scheduler,
    )

    best_validation = float("inf")
    total_loss = 0.0
    micro_steps = 0
    updates = 0
    started = monotonic()
    interval_loss = 0.0
    interval_steps = 0
    accelerator.print(
        json.dumps(
            {
                "event": "started",
                "train_samples": len(train_data),
                "validation_samples": len(validation_data),
                "planned_updates": total_updates,
            }
        ),
        flush=True,
    )
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        covered_samples = set()
        source_draws = Counter()
        half_checkpoint_saved = False
        for batch_index, batch in enumerate(train_loader, start=1):
            sample_ids = batch.pop("sample_ids")
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    logits_to_keep=1,
                )
                loss, _ = masked_action_cross_entropy(
                    outputs.logits,
                    batch["prediction_indices"],
                    batch["valid_action_token_ids"],
                    batch["valid_action_mask"],
                    batch["target_action_token_ids"],
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite SFT loss")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            covered_samples.update(sample_ids)
            source_draws.update(source_by_id[sample_id] for sample_id in sample_ids)
            total_loss += float(loss.detach())
            micro_steps += 1
            interval_loss += float(loss.detach())
            interval_steps += 1
            if accelerator.sync_gradients:
                updates += 1
                progress = {
                    "event": "update",
                    "epoch": epoch + 1,
                    "update": updates,
                    "planned_updates": total_updates,
                    "micro_steps": micro_steps,
                    "loss": interval_loss / interval_steps,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": monotonic() - started,
                }
                if accelerator.is_main_process:
                    write_json_atomic(Path(config.output_dir) / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                interval_loss = 0.0
                interval_steps = 0
                if batch_index in boundaries:
                    segment = boundaries.index(batch_index) + 1
                    checkpoint = Path(config.output_dir) / f"checkpoint-epoch-{epoch + 1}-part-{segment}"
                    _save_training_checkpoint(
                        accelerator, model, tokenizer, optimizer, scheduler, checkpoint,
                        config, epoch + 1, batch_index, updates, covered_samples,
                    )
                    if segment_callback is not None:
                        segment_callback(segment, checkpoint)
                if (
                    half_epoch_callback is not None
                    and not half_checkpoint_saved
                    and batch_index >= math.ceil(len(train_loader) / 2)
                    and batch_index < len(train_loader)
                ):
                    checkpoint = Path(config.output_dir) / f"checkpoint-epoch-{epoch + 1}-half"
                    _save_training_checkpoint(
                        accelerator, model, tokenizer, optimizer, scheduler, checkpoint,
                        config, epoch + 1, batch_index, updates, covered_samples,
                    )
                    half_checkpoint_saved = True
                    model.eval()
                    with torch.random.fork_rng():
                        half_epoch_callback(
                            epoch + 1, accelerator.unwrap_model(model), tokenizer, checkpoint
                        )
                    model.train()

        expected_samples = {row["sample_id"] for row in train_data.rows}
        missing_samples = sorted(expected_samples - covered_samples)
        coverage = {
            "epoch": epoch + 1,
            "expected_samples": len(expected_samples),
            "covered_samples": len(covered_samples),
            "missing_samples": missing_samples,
            "source_draws": dict(source_draws),
            "expected_source_draws": sampler.source_quotas(),
            "covered_sample_ids": sorted(covered_samples),
        }
        if accelerator.is_main_process:
            write_json_atomic(
                Path(config.output_dir) / f"coverage-epoch-{epoch + 1}.json", coverage
            )
        if config.cover_all_samples and missing_samples:
            raise RuntimeError(f"SFT coverage check failed: {len(missing_samples)} missing samples")
        if accelerator.num_processes == 1 and source_draws != Counter(sampler.source_quotas()):
            raise RuntimeError("SFT source sampling did not match the epoch plan")
        validation = _validate(model, validation_loader, accelerator)
        accelerator.print(
            json.dumps(
                {
                    "event": "validation",
                    "epoch": epoch + 1,
                    **validation,
                }
            ),
            flush=True,
        )
        if validation["validation_loss"] < best_validation:
            best_validation = validation["validation_loss"]
            if not segments:
                _save_checkpoint(
                    accelerator,
                    model,
                    tokenizer,
                    Path(config.output_dir) / "checkpoint-best",
                )
        if epoch_callback is not None:
            checkpoint = Path(config.output_dir) / f"checkpoint-epoch-{epoch + 1}"
            _save_training_checkpoint(
                accelerator, model, tokenizer, optimizer, scheduler, checkpoint,
                config, epoch + 1, len(train_loader), updates, covered_samples,
            )
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.random.fork_rng():
                epoch_callback(epoch + 1, accelerator.unwrap_model(model), tokenizer, checkpoint)
            model.train()

    metrics = {
        "train_loss": total_loss / max(micro_steps, 1),
        **_validate(model, validation_loader, accelerator),
        "best_validation_loss": best_validation,
    }
    if accelerator.is_main_process:
        write_json_atomic(
            Path(config.output_dir) / "training_manifest.json",
            {"config": asdict(config), "metrics": metrics},
        )
    accelerator.wait_for_everyone()
    return metrics


def _validate(model, loader, accelerator) -> dict[str, float]:
    import torch

    model.eval()
    totals = torch.zeros(3, device=accelerator.device)
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                logits_to_keep=1,
            )
            loss, logprobs = masked_action_cross_entropy(
                outputs.logits,
                batch["prediction_indices"],
                batch["valid_action_token_ids"],
                batch["valid_action_mask"],
                batch["target_action_token_ids"],
            )
            predicted_positions = logprobs.argmax(dim=1, keepdim=True)
            predicted_tokens = (
                batch["valid_action_token_ids"].gather(1, predicted_positions).squeeze(1)
            )
            batch_size = batch["input_ids"].shape[0]
            totals += torch.tensor(
                [
                    float(loss) * batch_size,
                    float((predicted_tokens == batch["target_action_token_ids"]).sum()),
                    batch_size,
                ],
                device=totals.device,
            )
    totals = accelerator.reduce(totals, reduction="sum")
    model.train()
    count = max(float(totals[2]), 1.0)
    return {
        "validation_loss": float(totals[0]) / count,
        "validation_action_accuracy": float(totals[1]) / count,
    }


def _save_checkpoint(accelerator, model, tokenizer, path: Path) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        path,
        state_dict=accelerator.get_state_dict(model),
        save_function=accelerator.save,
    )
    tokenizer.save_pretrained(path)


def _save_training_checkpoint(
    accelerator, model, tokenizer, optimizer, scheduler, path,
    config, epoch, batches_consumed, updates, covered_samples,
) -> None:
    import torch

    _save_checkpoint(accelerator, model, tokenizer, path)
    if not accelerator.is_main_process:
        return
    position = {
        "epoch": epoch,
        "batches_consumed": batches_consumed,
        "updates": updates,
        "covered_samples": len(covered_samples),
        "covered_sample_ids": sorted(covered_samples),
    }
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
            "position": position,
        },
        path / "training-state.pt",
    )
    write_json_atomic(path / "training-position.json", {**position, "config": asdict(config)})
    print(json.dumps({"event": "checkpoint_saved", "path": str(path), **position}
                     | {"covered_sample_ids": None}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(train(SFTTrainConfig.from_json(args.config)), sort_keys=True))


if __name__ == "__main__":
    main()
