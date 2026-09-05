"""Single-GPU GRPO-style LoRA update from verified online scheduler rollouts."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from tradingagents.scheduler.store import write_json_atomic

from .grpo_dataset import GRPOCollator, SchedulerGRPODataset
from .grpo_loss import grpo_clipped_loss, selected_action_logprobs
from .model import SchedulerModelConfig, load_scheduler_model


@dataclass(frozen=True)
class GRPOTrainConfig:
    rollout_path: str
    sft_adapter_path: str
    output_dir: str
    base_model: str = "Qwen/Qwen3-1.7B"
    base_revision: str | None = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    max_length: int = 32768
    rollout_temperature: float = 0.8
    group_size: int | None = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> GRPOTrainConfig:
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
    model, tokenizer, _ = load_scheduler_model(
        SchedulerModelConfig(
            base_model=config.base_model,
            base_revision=config.base_revision,
            adapter_path=config.sft_adapter_path,
            dtype=config.dtype,
            attention_implementation=config.attention_implementation,
        ),
        training=True,
    )
    dataset = SchedulerGRPODataset(
        config.rollout_path,
        expected_group_size=config.group_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        collate_fn=GRPOCollator(tokenizer, max_length=config.max_length),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    totals = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0, "clip_fraction": 0.0}
    steps = 0
    updates = 0
    model.train()
    # Rollout probabilities are measured without dropout; optimize the same policy.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for _ in range(config.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    logits_to_keep=1,
                )
                new_logprobs = selected_action_logprobs(
                    outputs.logits,
                    batch["prediction_indices"],
                    batch["valid_action_token_ids"],
                    batch["valid_action_mask"],
                    batch["target_action_token_ids"],
                    temperature=config.rollout_temperature,
                )
                loss_output = grpo_clipped_loss(
                    new_logprobs,
                    batch["old_logprobs"],
                    batch["ref_logprobs"],
                    batch["advantages"],
                    clip_epsilon=config.clip_epsilon,
                    kl_beta=config.kl_beta,
                    loss_weights=batch["loss_weights"],
                )
                if not bool(torch.isfinite(loss_output.loss)):
                    raise FloatingPointError("non-finite GRPO loss")
                accelerator.backward(loss_output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                updates += 1
                progress = {"update": updates, "micro_steps": steps + 1,
                            "loss": float(loss_output.loss.detach()),
                            "policy_loss": float(loss_output.policy_loss.detach()),
                            "kl": float(loss_output.kl.detach())}
                if accelerator.is_main_process:
                    write_json_atomic(Path(config.output_dir) / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            for name in totals:
                totals[name] += float(getattr(loss_output, name).detach())
            steps += 1

    metrics = {name: value / max(steps, 1) for name, value in totals.items()}
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        destination = Path(config.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            destination,
            state_dict=accelerator.get_state_dict(model),
            save_function=accelerator.save,
        )
        tokenizer.save_pretrained(destination)
        torch.save({"optimizer": optimizer.state_dict(), "updates": updates,
                    "micro_steps": steps, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()},
                   destination / "training-state.pt")
        write_json_atomic(
            destination / "training_manifest.json",
            {"config": asdict(config), "metrics": metrics},
        )
    return metrics


def main() -> None:
    from .gpu_lease import gpu_lease, set_gpu_budget

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-budget-gib", type=float, default=14.0)
    args = parser.parse_args()
    with gpu_lease():
        set_gpu_budget(args.gpu_budget_gib)
        print(json.dumps({"event": "gpu_budget", "gib": args.gpu_budget_gib}), flush=True)
        print(json.dumps(train(GRPOTrainConfig.from_json(args.config)), sort_keys=True))


if __name__ == "__main__":
    main()
