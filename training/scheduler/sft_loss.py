"""Masked next-action classification shared by SFT and runtime evaluation."""

from __future__ import annotations


def masked_action_logprobs(
    logits,
    prediction_indices,
    valid_action_token_ids,
    valid_action_mask,
    *,
    temperature: float = 1.0,
):
    import torch

    if temperature <= 0:
        raise ValueError("logprob temperature must be positive")
    if logits.shape[1] == 1:
        prediction_indices = torch.zeros_like(prediction_indices)
    batch = torch.arange(logits.shape[0], device=logits.device)
    decision_logits = logits[batch, prediction_indices]
    valid_logits = decision_logits.gather(1, valid_action_token_ids)
    valid_logits = valid_logits.masked_fill(~valid_action_mask, float("-inf"))
    return torch.log_softmax(valid_logits / temperature, dim=-1)


def masked_action_cross_entropy(
    logits,
    prediction_indices,
    valid_action_token_ids,
    valid_action_mask,
    target_action_token_ids,
):
    logprobs = masked_action_logprobs(
        logits,
        prediction_indices,
        valid_action_token_ids,
        valid_action_mask,
    )
    target_positions = valid_action_token_ids.eq(
        target_action_token_ids.unsqueeze(-1)
    ) & valid_action_mask
    if not bool(target_positions.any(dim=1).all()):
        raise ValueError("target action token is absent from the valid action mask")
    selected = logprobs.masked_fill(~target_positions, 0.0).sum(dim=1)
    return -selected.mean(), logprobs
