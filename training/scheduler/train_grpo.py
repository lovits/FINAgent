"""Offline GRPO-style LoRA update over verified grouped rollouts."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .grpo_dataset import GRPOCollator, GRPORolloutDataset
from .grpo_loss import grpo_clipped_loss, masked_action_logprobs
from .model import SchedulerModelConfig, load_scheduler_model


@dataclass(frozen=True)
class GRPOTrainConfig:
    base_model: str
    sft_adapter_path: str
    rollout_path: str
    output_dir: str
    base_model_revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "bfloat16"
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    learning_rate: float = 1e-5
    max_length: int = 4096
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def train(config: GRPOTrainConfig) -> dict[str, float]:
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
            adapter_path=config.sft_adapter_path,
            base_model_revision=config.base_model_revision,
            tokenizer_revision=config.tokenizer_revision,
            dtype=config.dtype,
        )
    )
    loader = DataLoader(
        GRPORolloutDataset(config.rollout_path),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=GRPOCollator(tokenizer, max_length=config.max_length),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    totals = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "clip_fraction": 0.0}
    steps = 0
    for _ in range(config.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                new_logprobs = masked_action_logprobs(
                    outputs.logits,
                    batch["prediction_indices"],
                    batch["valid_action_token_ids"],
                    batch["valid_action_mask"],
                    batch["selected_action_token_ids"],
                ).unsqueeze(-1)
                old_logprobs = batch["old_logprobs"].unsqueeze(-1)
                ref_logprobs = batch["ref_logprobs"].unsqueeze(-1)
                loss_output = grpo_clipped_loss(
                    new_logprobs,
                    old_logprobs,
                    ref_logprobs,
                    batch["advantages"],
                    torch.ones_like(new_logprobs, dtype=torch.bool),
                    clip_epsilon=config.clip_epsilon,
                    kl_beta=config.kl_beta,
                )
                accelerator.backward(loss_output.loss)
                optimizer.step()
                optimizer.zero_grad()
            for key in totals:
                value = getattr(loss_output, key if key != "loss" else "loss")
                totals[key] += float(value.detach())
            steps += 1

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
            json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
        )
    return {key: value / max(steps, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = train(GRPOTrainConfig.from_json(args.config))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
