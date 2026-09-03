from types import SimpleNamespace

import pytest
import torch

from tests.scheduler_helpers import FakeTokenizer
from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import SchedulerContext
from tradingagents.scheduler.hf_policy import HFSchedulerPolicy
from training.scheduler.model import register_action_tokens


class _Model:
    def __init__(self, tokenizer: FakeTokenizer):
        self.tokenizer = tokenizer

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, **kwargs):
        logits = torch.zeros((1, input_ids.shape[1], len(self.tokenizer)))
        logits[0, -1, self.tokenizer.action_ids[SchedulerAction.MARKET]] = 1.0
        logits[0, -1, self.tokenizer.action_ids[SchedulerAction.NEWS]] = 4.0
        return SimpleNamespace(logits=logits)


def _context() -> SchedulerContext:
    return SchedulerContext(
        "task-1",
        {},
        "choose",
        (SchedulerAction.MARKET, SchedulerAction.NEWS),
        ("market", "news"),
    )


def test_hf_policy_selects_only_from_valid_action_logits() -> None:
    tokenizer = FakeTokenizer()
    register_action_tokens(tokenizer)
    policy = HFSchedulerPolicy(
        _Model(tokenizer), tokenizer, policy_id="tiny", device="cpu"
    )

    decision = policy.select_action(_context())
    distribution = policy.action_logprobs(_context())
    assert decision.action is SchedulerAction.NEWS
    assert set(distribution) == {SchedulerAction.MARKET, SchedulerAction.NEWS}
    assert sum(torch.exp(torch.tensor(list(distribution.values())))).item() == pytest.approx(1)


def test_hf_policy_rejects_context_overflow() -> None:
    tokenizer = FakeTokenizer()
    register_action_tokens(tokenizer)
    policy = HFSchedulerPolicy(
        _Model(tokenizer),
        tokenizer,
        policy_id="tiny",
        device="cpu",
        max_context_tokens=3,
    )
    with pytest.raises(ValueError, match="maximum"):
        policy.select_action(_context())
