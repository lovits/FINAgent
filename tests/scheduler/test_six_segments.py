import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.scheduler_helpers import FakeTokenizer
from tests.scheduler.test_train_sft import _row, _tiny_qwen, _write_dataset
from training.scheduler.train_sft import SFTTrainConfig, checkpoint_boundaries, train
from training.scheduler.train_six_segments import publish_checkpoint, run, evaluate_queue


def test_boundaries_align_with_real_accumulation_and_include_last_partial_batch():
    assert checkpoint_boundaries(1683, 16, 6) == [288, 576, 848, 1136, 1408, 1683]
    with pytest.raises(ValueError, match="not enough"):
        checkpoint_boundaries(6, 2, 6)


def test_six_independent_checkpoints_cover_one_epoch_and_publish_after_save(monkeypatch, tmp_path):
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: FakeTokenizer())
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: _tiny_qwen())
    training = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_dataset(training, [_row(i, "static") for i in range(13)])
    _write_dataset(validation, [_row(99, "static")])
    output = tmp_path / "output"
    positions = []

    def callback(segment, checkpoint):
        assert (checkpoint / "adapter_model.safetensors").exists()
        state = torch.load(checkpoint / "training-state.pt", weights_only=False)
        positions.append(state["position"]["covered_samples"])
        publish_checkpoint(output, segment, checkpoint)

    train(SFTTrainConfig(
        train_path=str(training), validation_path=str(validation), output_dir=str(output),
        base_model="tiny", base_revision=None, dtype="float32", max_length=64,
        epochs=1, gradient_accumulation_steps=2,
    ), segments=6, segment_callback=callback)
    assert positions == [4, 6, 8, 10, 12, 13]
    assert len(list(output.glob("checkpoint-epoch-1-part-*"))) == 6
    coverage = json.loads((output / "coverage-epoch-1.json").read_text())
    assert coverage["covered_samples"] == 13
    assert coverage["source_draws"] == {"static": 13}
    assert not coverage["missing_samples"]


def test_worker_does_not_block_training_and_is_hidden_from_gpu(monkeypatch, tmp_path):
    events = []
    output = tmp_path / "output"

    class Worker:
        pid = 123

        def __init__(self, command, *, env, **kwargs):
            assert command[2] == "training.scheduler.train_six_segments"
            assert env["CUDA_VISIBLE_DEVICES"] == ""
            events.append("worker_started")

        def poll(self):
            return None

        def wait(self):
            events.append("wait_after_training")
            return 0

    def fake_train(config, **kwargs):
        assert kwargs["segments"] == 6
        events.append("train_completed")
        return {"train_loss": 0.5}

    monkeypatch.setattr("training.scheduler.train_six_segments.subprocess.Popen", Worker)
    monkeypatch.setattr("training.scheduler.train_six_segments.train", fake_train)
    run(SimpleNamespace(output_dir=str(output), epochs=1, cover_all_samples=True), "config", "tasks")
    assert events == ["worker_started", "train_completed", "wait_after_training"]
    assert json.loads((output / "cycle-complete.json").read_text())["evaluation_tasks"] == 90


def test_worker_evaluates_all_six_saved_models_with_teacher(monkeypatch, tmp_path):
    calls = []
    queue = tmp_path / "evaluation-queue"
    queue.mkdir()
    for part in range(1, 7):
        (queue / f"part-{part}.json").write_text(json.dumps({"checkpoint": f"checkpoint-{part}"}))
    monkeypatch.setattr(
        "training.scheduler.train_six_segments.load_scheduler_model",
        lambda *a, **kw: (SimpleNamespace(config=SimpleNamespace()), None, None),
    )

    def evaluate(*args, **kwargs):
        assert kwargs["include_teacher"]
        calls.append((str(args[3]), kwargs["stage_name"]))

    monkeypatch.setattr("training.scheduler.train_six_segments.evaluate_epoch", evaluate)
    evaluate_queue(SimpleNamespace(
        output_dir=str(tmp_path), base_model="base", base_revision=None,
        dtype="bfloat16", attention_implementation="sdpa", max_length=32768,
    ), [])
    assert calls == [(f"checkpoint-{part}", f"part-{part}") for part in range(1, 7)]
