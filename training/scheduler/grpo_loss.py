"""Token-masked GRPO-style objective for scheduler actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GRPOLossOutput:
    loss: Any
    policy_loss: Any
    kl: Any
    clip_fraction: Any


def token_logprobs(logits, input_ids):
    """Return next-token log probabilities aligned to input_ids[:, 1:]."""

    import torch

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    return log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def masked_action_logprobs(
    logits,
    prediction_indices,
    valid_action_token_ids,
    valid_action_mask,
    selected_action_token_ids,
):
    """Score selected actions under the same valid-action mask used for sampling."""

    import torch

    batch = torch.arange(logits.shape[0], device=logits.device)
    decision_logits = logits[batch, prediction_indices]
    valid_logits = decision_logits.gather(1, valid_action_token_ids)
    valid_logits = valid_logits.masked_fill(~valid_action_mask, float("-inf"))
    log_probabilities = torch.log_softmax(valid_logits, dim=-1)
    selected_positions = valid_action_token_ids.eq(
        selected_action_token_ids.unsqueeze(-1)
    ) & valid_action_mask
    if not bool(selected_positions.any(dim=1).all()):
        raise ValueError("selected action token is missing from valid action mask")
    return log_probabilities.masked_fill(~selected_positions, 0.0).sum(dim=1)


def grpo_clipped_loss(
    new_logprobs,
    old_logprobs,
    ref_logprobs,
    advantages,
    action_mask,
    *,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.01,
) -> GRPOLossOutput:
    """Compute clipped policy loss and reference KL on action tokens only."""

    import torch

    mask = action_mask.to(dtype=new_logprobs.dtype)
    denominator = mask.sum()
    if denominator.item() <= 0:
        raise ValueError("GRPO batch contains no scheduler action tokens")
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)

    ratio = torch.exp(new_logprobs - old_logprobs.detach())
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    clipped = clipped_ratio * advantages
    policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / denominator

    ref_delta = ref_logprobs.detach() - new_logprobs
    per_token_kl = torch.exp(ref_delta) - ref_delta - 1
    kl = (per_token_kl * mask).sum() / denominator
    loss = policy_loss + kl_beta * kl
    clip_fraction = (
        ((ratio - 1).abs() > clip_epsilon).to(mask.dtype) * mask
    ).sum() / denominator
    return GRPOLossOutput(loss, policy_loss, kl, clip_fraction)
