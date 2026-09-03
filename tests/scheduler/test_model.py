import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from tests.scheduler_helpers import FakeTokenizer
from tradingagents.scheduler.actions import SchedulerAction
from training.scheduler.model import SchedulerModelConfig, load_scheduler_model
from training.scheduler.sft_dataset import MaskedActionCollator
from training.scheduler.sft_loss import masked_action_cross_entropy


def _tiny_qwen() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            tie_word_embeddings=True,
        )
    )


def test_loader_adds_action_rows_and_lora_without_full_embedding_training(
    monkeypatch,
) -> None:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _tiny_qwen(),
    )

    model, loaded_tokenizer, action_ids = load_scheduler_model(
        SchedulerModelConfig(base_model="tiny", base_revision=None),
        training=True,
    )

    assert loaded_tokenizer is tokenizer
    assert len(action_ids) == 13
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    assert input_embeddings.token_adapter.base_layer.num_embeddings == 77
    assert output_embeddings.token_adapter.tied_adapter is input_embeddings.token_adapter
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    assert any("lora_" in name for name in trainable_names)
    assert any("trainable_tokens" in name for name in trainable_names)
    assert not any(name.endswith("original_module.weight") for name in trainable_names)


def test_tiny_qwen_completes_masked_action_backward(monkeypatch) -> None:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _tiny_qwen(),
    )
    model, tokenizer, _ = load_scheduler_model(
        SchedulerModelConfig(base_model="tiny", base_revision=None),
        training=True,
    )
    batch = MaskedActionCollator(tokenizer, max_length=64)(
        [
            {
                "sample_id": "sample-1",
                "input_text": "choose next",
                "valid_actions": ["<ACT_MARKET>", "<ACT_NEWS>"],
                "target_action": "<ACT_NEWS>",
            }
        ]
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.1,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    loss, _ = masked_action_cross_entropy(
        outputs.logits,
        batch["prediction_indices"],
        batch["valid_action_token_ids"],
        batch["valid_action_mask"],
        batch["target_action_token_ids"],
    )
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert tokenizer.action_ids[SchedulerAction.NEWS] in batch["valid_action_token_ids"]
