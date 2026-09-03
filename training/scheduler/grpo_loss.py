"""Clipped group-relative policy loss over one scheduler action per row."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sft_loss import masked_action_logprobs


@dataclass(frozen=True)
class GRPOLossOutput:
    loss: Any
    policy_loss: Any
    kl: Any
    clip_fraction: Any


def selected_action_logprobs(
    logits,
    prediction_indices,
    valid_action_token_ids,
    valid_action_mask,
    selected_action_token_ids,
    *,
    temperature: float = 1.0,
):
    logprobs = masked_action_logprobs(
        logits,
        prediction_indices,
        valid_action_token_ids,
        valid_action_mask,
        temperature=temperature,
    )
    selected = valid_action_token_ids.eq(
        selected_action_token_ids.unsqueeze(-1)
    ) & valid_action_mask
    if not bool(selected.any(dim=1).all()):
        raise ValueError("selected action is absent from valid action mask")
    return logprobs.masked_fill(~selected, 0.0).sum(dim=1)


def grpo_clipped_loss(
    new_logprobs,
    old_logprobs,
    ref_logprobs,
    advantages,
    *,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.01,
) -> GRPOLossOutput:
    import torch

    ratio = torch.exp(new_logprobs - old_logprobs.detach())
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    clipped = clipped_ratio * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    ref_delta = ref_logprobs.detach() - new_logprobs
    kl = (torch.exp(ref_delta) - ref_delta - 1).mean()
    loss = policy_loss + kl_beta * kl
    clip_fraction = ((ratio - 1).abs() > clip_epsilon).float().mean()
    return GRPOLossOutput(loss, policy_loss, kl, clip_fraction)
