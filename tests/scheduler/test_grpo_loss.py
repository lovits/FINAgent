import torch

from training.scheduler.grpo_loss import (
    grpo_clipped_loss,
    selected_action_logprobs,
)


def test_selected_logprobs_use_only_valid_action_tokens() -> None:
    logits = torch.zeros((1, 2, 8))
    logits[0, 1, 2] = 1.0
    logits[0, 1, 3] = 3.0
    logits[0, 1, 7] = 100.0
    selected = selected_action_logprobs(
        logits,
        torch.tensor([1]),
        torch.tensor([[2, 3]]),
        torch.tensor([[True, True]]),
        torch.tensor([3]),
    )
    assert selected.item() > -0.2


def test_grpo_loss_backpropagates_and_clips_ratio() -> None:
    new = torch.tensor([-0.2, -1.0], requires_grad=True)
    old = torch.tensor([-0.5, -0.5])
    reference = torch.tensor([-0.6, -0.6])
    advantages = torch.tensor([1.0, -1.0])
    output = grpo_clipped_loss(new, old, reference, advantages)
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert new.grad is not None
    assert output.clip_fraction.item() > 0
