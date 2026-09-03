"""AutoDL-ready LoRA SFT entry point for scheduler action prediction."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .collator import ActionOnlyCollator
from .dataset import SchedulerSFTDataset
from .model import SchedulerModelConfig, load_scheduler_model


@dataclass(frozen=True)
class SFTTrainConfig:
    base_model: str
    train_path: str
    output_dir: str
    validation_path: str | None = None
    base_model_revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "bfloat16"
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: int = 2
    learning_rate: float = 2e-5
    max_length: int = 4096
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _validation_loss(model, loader, accelerator) -> float:
    import torch

    model.eval()
    loss_sum = torch.zeros((), device=accelerator.device)
    example_count = torch.zeros((), device=accelerator.device)
    with torch.no_grad():
        for batch in loader:
            batch_size = batch["input_ids"].shape[0]
            loss_sum += model(**batch).loss.detach() * batch_size
            example_count += batch_size
    totals = accelerator.reduce(
        torch.stack((loss_sum, example_count)),
        reduction="sum",
    )
    model.train()
    return float((totals[0] / totals[1].clamp_min(1)).item())


def train(config: SFTTrainConfig) -> dict[str, float]:
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    model, tokenizer = load_scheduler_model(
        SchedulerModelConfig(
            base_model=config.base_model,
            base_model_revision=config.base_model_revision,
            tokenizer_revision=config.tokenizer_revision,
            dtype=config.dtype,
        )
    )
    collator = ActionOnlyCollator(tokenizer, max_length=config.max_length)
    loader = DataLoader(
        SchedulerSFTDataset(config.train_path),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = None
    if config.validation_path:
        validation_loader = DataLoader(
            SchedulerSFTDataset(config.validation_path),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    if validation_loader is None:
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    else:
        model, optimizer, loader, validation_loader = accelerator.prepare(
            model,
            optimizer,
            loader,
            validation_loader,
        )
    model.train()
    total_loss = 0.0
    update_steps = 0
    for _ in range(config.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += float(loss.detach())
            update_steps += 1

    metrics = {"train_loss": total_loss / max(update_steps, 1)}
    if validation_loader is not None:
        metrics["validation_loss"] = _validation_loss(
            model, validation_loader, accelerator
        )

    accelerator.wait_for_everyone()
    output = Path(config.output_dir)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            output,
            state_dict=accelerator.get_state_dict(model),
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(output)
        (output / "training_manifest.json").write_text(
            json.dumps(
                {"config": asdict(config), "metrics": metrics},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = train(SFTTrainConfig.from_json(args.config))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
