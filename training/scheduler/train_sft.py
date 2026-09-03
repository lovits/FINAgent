"""Single-GPU masked-action LoRA SFT for the scheduler policy."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from tradingagents.scheduler.store import write_json_atomic

from .model import SchedulerModelConfig, load_scheduler_model
from .sft_dataset import (
    HierarchicalSourceSampler,
    MaskedActionCollator,
    SchedulerSFTDataset,
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
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> SFTTrainConfig:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def train(config: SFTTrainConfig) -> dict[str, float]:
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
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
    train_data = SchedulerSFTDataset(config.train_path)
    validation_data = SchedulerSFTDataset(config.validation_path)
    sampler = HierarchicalSourceSampler(
        train_data.rows,
        total_epochs=config.epochs,
        seed=config.seed,
    )
    collator = MaskedActionCollator(tokenizer, max_length=config.max_length)
    train_loader = DataLoader(
        train_data,
        batch_size=config.micro_batch_size,
        sampler=sampler,
        collate_fn=collator,
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
    updates_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_updates = max(1, updates_per_epoch * config.epochs)
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
    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
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
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            total_loss += float(loss.detach())
            micro_steps += 1

        validation = _validate(model, validation_loader, accelerator)
        if validation["validation_loss"] < best_validation:
            best_validation = validation["validation_loss"]
            _save_checkpoint(
                accelerator,
                model,
                tokenizer,
                Path(config.output_dir) / "checkpoint-best",
            )

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
            predicted_tokens = batch["valid_action_token_ids"].gather(
                1, predicted_positions
            ).squeeze(1)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(train(SFTTrainConfig.from_json(args.config)), sort_keys=True))


if __name__ == "__main__":
    main()
