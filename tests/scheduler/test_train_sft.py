import json

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from tests.scheduler_helpers import FakeTokenizer
from training.scheduler.model import SchedulerModelConfig, load_scheduler_model
from training.scheduler.train_sft import SFTTrainConfig, train


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


def _write_dataset(path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(index: int, source: str) -> dict:
    return {
        "schema_version": "scheduler-sft-v1",
        "sample_id": f"sample-{index}",
        "task_id": f"task-{index}",
        "trajectory_id": f"trajectory-{index}",
        "step_id": 0,
        "source": source,
        "input_text": f"choose action {index}",
        "valid_actions": ["<ACT_MARKET>", "<ACT_NEWS>"],
        "target_action": "<ACT_NEWS>",
        "input_token_count": 16,
    }


def test_tiny_sft_entry_point_runs_update_validation_and_checkpoint(
    monkeypatch, tmp_path
) -> None:
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
        "training.scheduler.train_sft.load_scheduler_model",
        lambda *args, **kwargs: tiny_model,
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    _write_dataset(
        train_path,
        [_row(0, "static"), _row(1, "teacher_verified")],
    )
    _write_dataset(validation_path, [_row(2, "static")])

    metrics = train(
        SFTTrainConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
            output_dir=str(tmp_path / "output"),
            base_model="tiny",
            base_revision=None,
            dtype="float32",
            max_length=64,
            gradient_accumulation_steps=1,
            epochs=1,
        )
    )

    assert 0 <= metrics["validation_action_accuracy"] <= 1
    assert (tmp_path / "output" / "checkpoint-best" / "adapter_config.json").exists()
    manifest = json.loads(
        (tmp_path / "output" / "training_manifest.json").read_text()
    )
    assert manifest["config"]["base_model"] == "tiny"
