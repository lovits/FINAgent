"""Local Hugging Face scheduler policy over a hard-masked action vocabulary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .actions import SchedulerAction
from .contracts import PolicyDecision, SchedulerContext


class HFSchedulerPolicy:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        policy_id: str,
        device: str,
        temperature: float = 0.0,
        max_context_tokens: int = 32768,
        adapter_name: str | None = None,
    ):
        if temperature < 0:
            raise ValueError("scheduler temperature cannot be negative")
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.policy_id = policy_id
        self.device = device
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self.adapter_name = adapter_name

    def action_logprobs(
        self, context: SchedulerContext
    ) -> Mapping[SchedulerAction, float]:
        values, _ = self._distribution(context)
        return {
            action: float(values[index].item())
            for index, action in enumerate(context.valid_actions)
        }

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        import torch

        logprobs, input_tokens = self._distribution(context)
        if self.temperature > 0:
            selected_index = int(torch.multinomial(torch.exp(logprobs), 1).item())
        else:
            selected_index = int(torch.argmax(logprobs).item())
        return PolicyDecision(
            context.valid_actions[selected_index],
            policy_id=self.policy_id,
            logprob=float(logprobs[selected_index].item()),
            metadata={
                "usage": {
                    "llm_calls": 1,
                    "input_tokens": input_tokens,
                    "output_tokens": 1,
                    "reported": True,
                    "source": "local_tokenizer",
                }
            },
        )

    def _distribution(self, context: SchedulerContext):
        import torch

        from training.scheduler.model import scheduler_input_ids

        if self.adapter_name is not None:
            if not hasattr(self.model, "set_adapter"):
                raise ValueError("adapter_name requires a PEFT model")
            self.model.set_adapter(self.adapter_name)
        input_ids = scheduler_input_ids(self.tokenizer, context.serialized_state)
        if len(input_ids) > self.max_context_tokens:
            raise ValueError(
                f"scheduler context has {len(input_ids)} tokens; "
                f"maximum is {self.max_context_tokens}"
            )
        tensor = torch.tensor([input_ids], device=self.device)
        with torch.no_grad():
            logits = self.model(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                logits_to_keep=1,
            ).logits[0, -1]
        token_ids = torch.tensor(
            [self._action_token_id(action) for action in context.valid_actions],
            device=logits.device,
        )
        valid_logits = logits[token_ids]
        scale = self.temperature if self.temperature > 0 else 1.0
        return torch.log_softmax(valid_logits / scale, dim=-1), len(input_ids)

    def _action_token_id(self, action: SchedulerAction) -> int:
        values = self.tokenizer(action.value, add_special_tokens=False)["input_ids"]
        if len(values) != 1:
            raise ValueError(f"scheduler action is not one token: {action.value}")
        return int(values[0])


def load_hf_scheduler_policy(config: dict[str, Any]) -> HFSchedulerPolicy:
    import torch

    from training.scheduler.model import SchedulerModelConfig, load_scheduler_model

    adapter_path = config.get("scheduler_adapter_path")
    if not adapter_path:
        raise ValueError("learned mode requires scheduler_adapter_path")
    model, tokenizer, _ = load_scheduler_model(
        SchedulerModelConfig(
            base_model=str(config.get("scheduler_base_model", "Qwen/Qwen3-1.7B")),
            base_revision=config.get("scheduler_base_revision"),
            adapter_path=str(adapter_path),
            dtype=str(config.get("scheduler_dtype", "bfloat16")),
        ),
        training=False,
    )
    device = config.get("scheduler_device") or ("cuda" if torch.cuda.is_available() else "cpu")
    return HFSchedulerPolicy(
        model,
        tokenizer,
        policy_id=f"hf:{adapter_path}",
        device=str(device),
        temperature=float(config.get("scheduler_action_temperature", 0.0)),
        max_context_tokens=int(config.get("scheduler_max_context_tokens", 32768)),
    )


def load_shared_hf_scheduler_policies(
    config: dict[str, Any],
    *,
    active_adapter_path: str,
    reference_adapter_path: str,
    temperature: float,
) -> tuple[HFSchedulerPolicy, HFSchedulerPolicy]:
    """Load one frozen Base and switch small active/reference adapters sequentially."""

    import torch

    from training.scheduler.model import SchedulerModelConfig, load_scheduler_model

    model, tokenizer, _ = load_scheduler_model(
        SchedulerModelConfig(
            base_model=str(config.get("scheduler_base_model", "Qwen/Qwen3-1.7B")),
            base_revision=config.get("scheduler_base_revision"),
            adapter_path=active_adapter_path,
            dtype=str(config.get("scheduler_dtype", "bfloat16")),
        ),
        training=False,
    )
    model.load_adapter(
        reference_adapter_path,
        adapter_name="reference",
        is_trainable=False,
    )
    device = str(
        config.get("scheduler_device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    common = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "temperature": temperature,
        "max_context_tokens": int(config.get("scheduler_max_context_tokens", 32768)),
    }
    active = HFSchedulerPolicy(
        **common,
        policy_id=f"hf:{active_adapter_path}",
        adapter_name="default",
    )
    reference = HFSchedulerPolicy(
        **common,
        policy_id=f"hf-ref:{reference_adapter_path}",
        adapter_name="reference",
    )
    return active, reference
