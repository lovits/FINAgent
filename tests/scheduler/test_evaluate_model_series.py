import json
from pathlib import Path
from threading import BoundedSemaphore
from types import SimpleNamespace

import torch

from training.scheduler.evaluate_model_series import (
    ConcurrentCheckpointPolicy, checkpoints,
)


def test_five_complete_rl_checkpoints_are_selected(tmp_path):
    for number in range(1, 6):
        checkpoint = tmp_path / f"round-{number}/checkpoint"
        checkpoint.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "training-state.pt",
                     "training_manifest.json"):
            (checkpoint / name).write_text("complete")
    assert list(checkpoints({"output_dir": str(tmp_path)})) == [
        "rl-1", "rl-2", "rl-3", "rl-4", "rl-5",
    ]


def test_each_model_serializes_itself_while_sharing_gpu_slots(monkeypatch):
    model = SimpleNamespace()
    model.to = lambda device: setattr(model, "device", device) or model
    model.eval = lambda: model
    policy = ConcurrentCheckpointPolicy(
        model, None, policy_id="test", device="cpu",
        gpu_slots=BoundedSemaphore(3),
    )

    def distribution(self, context):
        assert self.model_lock.locked()
        assert model.device == "cuda"
        return torch.tensor([0.0])

    monkeypatch.setattr("tradingagents.scheduler.hf_policy.HFSchedulerPolicy._distribution",
                        distribution)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    result = policy._distribution(SimpleNamespace(task_id="task"))
    assert result.device.type == "cpu" and model.device == "cpu"
