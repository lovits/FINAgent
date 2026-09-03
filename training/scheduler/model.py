"""Hugging Face and PEFT construction for the Qwen scheduler policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tradingagents.scheduler.actions import SchedulerAction


@dataclass(frozen=True)
class SchedulerModelConfig:
    base_model: str = "Qwen/Qwen3-1.7B"
    base_revision: str | None = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    tokenizer_revision: str | None = None
    adapter_path: str | None = None
    dtype: str = "bfloat16"
    attention_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


def _torch_dtype(torch: Any, name: str):
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported scheduler dtype: {name!r}") from exc


def register_action_tokens(tokenizer: Any) -> tuple[int, ...]:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [action.value for action in SchedulerAction]}
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = tuple(
        int(tokenizer.convert_tokens_to_ids(action.value)) for action in SchedulerAction
    )
    if len(set(token_ids)) != len(SchedulerAction):
        raise ValueError("scheduler action tokens must map to unique token ids")
    for action, token_id in zip(SchedulerAction, token_ids, strict=True):
        encoded = tokenizer(action.value, add_special_tokens=False)["input_ids"]
        if encoded != [token_id]:
            raise ValueError(f"scheduler action is not one token: {action.value}")
    return token_ids


def scheduler_input_ids(tokenizer: Any, input_text: str) -> list[int]:
    """Normalize old list and current BatchEncoding chat-template return types."""

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": input_text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("scheduler chat template returned more than one sequence")
        values = values[0]
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError("scheduler chat template returned invalid input_ids")
    return values


def load_scheduler_model(
    config: SchedulerModelConfig,
    *,
    training: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = config.adapter_path or config.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None
        if config.adapter_path
        else config.tokenizer_revision or config.base_revision,
    )
    action_token_ids = register_action_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        dtype=_torch_dtype(torch, config.dtype),
        attn_implementation=config.attention_implementation,
    )
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.use_cache = not training

    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            config.adapter_path,
            is_trainable=training,
        )
    elif training:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules),
            trainable_token_indices=list(action_token_ids),
            bias="none",
        )
        model = get_peft_model(model, lora)
    if training and config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer, action_token_ids
