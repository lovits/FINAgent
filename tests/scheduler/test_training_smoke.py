import json

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from training.scheduler.model import SchedulerModelConfig, load_scheduler_model
from training.scheduler.train_grpo import GRPOTrainConfig, train as train_grpo
from training.scheduler.train_sft import SFTTrainConfig, train as train_sft


def _tiny_local_model(path):
    tokenizer_backend = Tokenizer(
        models.WordLevel(
            vocab={"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "state": 3},
            unk_token="[UNK]",
        )
    )
    tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
    )
    model.save_pretrained(path)


def test_tiny_local_model_runs_lora_sft_and_grpo_without_download(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    _tiny_local_model(base)

    model, _ = load_scheduler_model(
        SchedulerModelConfig(base_model=str(base), dtype="float32")
    )
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all("lora_" in name or "modules_to_save" in name for name in trainable)

    sft_data = tmp_path / "sft.jsonl"
    sft_data.write_text(
        json.dumps({"input_text": "state", "target_action": "<ACT_MARKET>"}) + "\n",
        encoding="utf-8",
    )
    sft_output = tmp_path / "sft-adapter"
    sft_metrics = train_sft(
        SFTTrainConfig(
            base_model=str(base),
            train_path=str(sft_data),
            validation_path=str(sft_data),
            output_dir=str(sft_output),
            dtype="float32",
            batch_size=1,
            gradient_accumulation_steps=1,
            epochs=1,
            learning_rate=1e-3,
            max_length=32,
        )
    )
    assert sft_metrics["train_loss"] > 0
    assert sft_metrics["validation_loss"] > 0
    assert (sft_output / "adapter_config.json").exists()

    grpo_data = tmp_path / "grpo.jsonl"
    grpo_data.write_text(
        json.dumps(
            {
                "serialized_state": "state",
                "selected_action": "<ACT_MARKET>",
                "valid_actions": ["<ACT_MARKET>", "<ACT_NEWS>"],
                "advantage": 1.0,
                "old_logprob": -0.693147,
                "ref_logprob": -0.693147,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    grpo_output = tmp_path / "grpo-adapter"
    grpo_metrics = train_grpo(
        GRPOTrainConfig(
            base_model=str(base),
            sft_adapter_path=str(sft_output),
            rollout_path=str(grpo_data),
            output_dir=str(grpo_output),
            dtype="float32",
            batch_size=1,
            gradient_accumulation_steps=1,
            epochs=1,
            learning_rate=1e-3,
            max_length=32,
        )
    )
    assert set(grpo_metrics) == {"loss", "policy_loss", "kl", "clip_fraction"}
    assert (grpo_output / "adapter_config.json").exists()
