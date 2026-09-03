"""Lazy Hugging Face/PEFT model construction for AutoDL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradingagents.scheduler.actions import SchedulerAction


@dataclass(frozen=True)
class SchedulerModelConfig:
    base_model: str
    adapter_path: str | None = None
    base_model_revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    modules_to_save: tuple[str, ...] = ("embed_tokens", "lm_head")


def _torch_dtype(torch: Any, name: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name!r}") from exc


def load_scheduler_model(config: SchedulerModelConfig, *, training: bool = True):
    """Load a causal LM, register action tokens, and optionally attach LoRA."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = config.adapter_path or config.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None
        if config.adapter_path
        else config.tokenizer_revision or config.base_model_revision,
    )
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [action.value for action in SchedulerAction]}
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.base_model_revision,
        dtype=_torch_dtype(torch, config.dtype),
    )
    embedding_size = model.get_input_embeddings().num_embeddings
    if added or embedding_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, config.adapter_path, is_trainable=training
        )
    elif training:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            modules_to_save=list(config.modules_to_save),
            bias="none",
        )
        model = get_peft_model(model, lora)
    return model, tokenizer
