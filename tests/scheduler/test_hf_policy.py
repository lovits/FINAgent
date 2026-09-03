from types import SimpleNamespace

import torch

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.hf_policy import HFSchedulerPolicy
from tradingagents.scheduler.policy import SchedulerContext


class _Tokenizer:
    tokens = {
        "<ACT_MARKET>": 5,
        "<ACT_NEWS>": 7,
    }

    def __call__(self, text, **kwargs):
        if text in self.tokens:
            return {"input_ids": [self.tokens[text]]}
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class _Model:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        logits = torch.zeros((1, 3, 10))
        logits[0, -1, 5] = 1.0
        logits[0, -1, 7] = 4.0
        return SimpleNamespace(logits=logits)


def test_hf_policy_selects_only_from_valid_action_logits():
    policy = HFSchedulerPolicy(
        _Model(),
        _Tokenizer(),
        policy_id="adapter-v1",
        device="cpu",
    )
    context = SchedulerContext(
        state={},
        serialized_state="{}",
        valid_actions=(SchedulerAction.MARKET, SchedulerAction.NEWS),
        selected_analysts=("market", "news"),
        step=0,
    )

    decision = policy.select_action(context)

    assert decision.action is SchedulerAction.NEWS
    assert decision.logprob is not None
