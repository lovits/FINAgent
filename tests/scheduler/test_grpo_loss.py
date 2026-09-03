import pytest

torch = pytest.importorskip("torch")

from training.scheduler.grpo_loss import (  # noqa: E402
    grpo_clipped_loss,
    masked_action_logprobs,
)


def test_grpo_gradient_increases_good_action_and_decreases_bad_action():
    new = torch.tensor([[0.0, 9.0], [0.0, -9.0]], requires_grad=True)
    old = torch.zeros_like(new)
    ref = torch.zeros_like(new)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[True, False], [True, False]])

    output = grpo_clipped_loss(
        new,
        old,
        ref,
        advantages,
        mask,
        clip_epsilon=0.2,
        kl_beta=0.0,
    )
    output.loss.backward()

    assert new.grad[0, 0] < 0
    assert new.grad[1, 0] > 0
    assert new.grad[:, 1].equal(torch.zeros(2))


def test_grpo_rejects_batch_without_action_tokens():
    values = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="no scheduler action tokens"):
        grpo_clipped_loss(
            values,
            values,
            values,
            torch.ones(2),
            torch.zeros_like(values, dtype=torch.bool),
        )


def test_masked_action_logprob_ignores_invalid_vocabulary_logits():
    logits = torch.zeros((1, 2, 8))
    logits[0, 0, 1] = 2.0
    logits[0, 0, 2] = 1.0
    logits[0, 0, 7] = 100.0

    selected = masked_action_logprobs(
        logits,
        torch.tensor([0]),
        torch.tensor([[1, 2]]),
        torch.tensor([[True, True]]),
        torch.tensor([1]),
    )

    assert torch.allclose(selected, torch.log_softmax(torch.tensor([2.0, 1.0]), 0)[0:1])
