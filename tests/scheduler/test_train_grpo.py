import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from tests.scheduler_helpers import FakeTokenizer
from training.scheduler.model import SchedulerModelConfig, load_scheduler_model
from training.scheduler.train_grpo import GRPOTrainConfig, train


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


def _row(index: int, action: str, advantage: float) -> dict:
    return {
        "schema_version": "scheduler-grpo-v1",
        "rollout_group_id": "group-1",
        "trajectory_id": f"trajectory-{index}",
        "task_id": "task-1",
        "step_id": 0,
        "serialized_state": f"choose action {index}",
        "valid_actions": ["<ACT_MARKET>", "<ACT_NEWS>"],
        "selected_action": action,
        "old_logprob": -0.693,
        "ref_logprob": -0.693,
        "reward_total": advantage,
        "reward_components": {},
        "advantage": advantage,
    }


def test_tiny_grpo_entry_point_updates_and_saves_adapter(monkeypatch, tmp_path) -> None:
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _tiny_qwen(),
    )
    tiny_model = load_scheduler_model(
        SchedulerModelConfig(base_model="tiny", base_revision=None),
        training=True,
    )
    monkeypatch.setattr(
        "training.scheduler.train_grpo.load_scheduler_model",
        lambda *args, **kwargs: tiny_model,
    )
    rollout_path = tmp_path / "grpo.jsonl"
    with rollout_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_row(0, "<ACT_NEWS>", 1.0)) + "\n")
        handle.write(json.dumps(_row(1, "<ACT_MARKET>", -1.0)) + "\n")

    metrics = train(
        GRPOTrainConfig(
            rollout_path=str(rollout_path),
            sft_adapter_path="fake-sft-adapter",
            output_dir=str(tmp_path / "output"),
            base_model="tiny",
            base_revision=None,
            dtype="float32",
            max_length=64,
            gradient_accumulation_steps=1,
            epochs=1,
        )
    )

    assert set(metrics) == {"loss", "policy_loss", "kl", "clip_fraction"}
    assert (tmp_path / "output" / "adapter_config.json").exists()
    manifest = json.loads(
        (tmp_path / "output" / "training_manifest.json").read_text()
    )
    assert manifest["config"]["rollout_temperature"] == 0.8
