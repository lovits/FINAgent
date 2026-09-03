"""Optional Hugging Face runtime policy for a trained scheduler adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .actions import SchedulerAction, parse_action
from .policy import PolicyDecision, SchedulerContext


def resolve_torch_device(torch: Any, requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFSchedulerPolicy:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        policy_id: str,
        device: str,
        temperature: float = 0.0,
        max_context_tokens: int = 4096,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.policy_id = policy_id
        self.device = device
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self.model.to(device)
        self.model.eval()

    def _action_token_id(self, action: SchedulerAction) -> int:
        token_ids = self.tokenizer(action.value, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"action is not a single tokenizer token: {action.value}")
        return int(token_ids[0])

    def _action_log_distribution(self, context: SchedulerContext):
        import torch

        prompt = context.serialized_state.rstrip() + "\n<SCHEDULER_ACTION>\n"
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits[0, -1]

        token_ids = torch.tensor(
            [self._action_token_id(action) for action in context.valid_actions],
            device=logits.device,
        )
        valid_logits = logits[token_ids]
        scale = self.temperature if self.temperature > 0 else 1.0
        return torch.log_softmax(valid_logits / scale, dim=-1)

    def action_logprob(
        self, context: SchedulerContext, action: SchedulerAction | str
    ) -> float:
        parsed = parse_action(action)
        if parsed not in context.valid_actions:
            raise ValueError(f"cannot score masked action: {parsed.value}")
        log_probabilities = self._action_log_distribution(context)
        index = context.valid_actions.index(parsed)
        return float(log_probabilities[index].item())

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        import torch

        log_probabilities = self._action_log_distribution(context)
        if self.temperature > 0:
            probabilities = torch.exp(log_probabilities)
            selected_index = int(torch.multinomial(probabilities, 1).item())
        else:
            selected_index = int(torch.argmax(log_probabilities).item())
        return PolicyDecision(
            action=context.valid_actions[selected_index],
            reason_code="HF_POLICY",
            logprob=float(log_probabilities[selected_index].item()),
        )


def load_hf_scheduler_policy(config: dict[str, Any]) -> HFSchedulerPolicy:
    """Load the configured base model plus trained adapter lazily."""

    base_model = config.get("scheduler_base_model")
    adapter_path = config.get("scheduler_adapter_path")
    if not base_model or not adapter_path:
        raise ValueError(
            "learned mode requires scheduler_base_model and scheduler_adapter_path"
        )
    if not Path(adapter_path).exists():
        raise ValueError(f"scheduler adapter path does not exist: {adapter_path}")

    import torch

    from training.scheduler.model import SchedulerModelConfig, load_scheduler_model

    model, tokenizer = load_scheduler_model(
        SchedulerModelConfig(
            base_model=str(base_model),
            adapter_path=str(adapter_path),
            base_model_revision=config.get("scheduler_base_model_revision"),
            tokenizer_revision=config.get("scheduler_tokenizer_revision"),
            dtype=config.get("scheduler_dtype", "bfloat16"),
        ),
        training=False,
    )
    device = resolve_torch_device(torch, config.get("scheduler_device"))
    return HFSchedulerPolicy(
        model,
        tokenizer,
        policy_id=f"hf:{Path(adapter_path).name}",
        device=device,
        temperature=float(config.get("scheduler_action_temperature", 0.0)),
        max_context_tokens=int(config.get("scheduler_max_context_tokens", 4096)),
    )
