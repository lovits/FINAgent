import json
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest
import torch

from training.scheduler.parallel_evaluation import SharedGPUModelPolicy, main


@pytest.mark.parametrize("fail", [False, True])
def test_gpu_forward_holds_shared_lock_and_always_offloads(monkeypatch, fail):
    lock = Lock()
    model = SimpleNamespace()
    model.to = lambda device: setattr(model, "device", device) or model
    model.eval = lambda: model

    def distribution(self, context):
        assert lock.locked() and model.device == self.device == "cuda"
        if fail:
            raise ValueError("inference failed")
        return torch.tensor([0.0])

    monkeypatch.setattr("tradingagents.scheduler.hf_policy.HFSchedulerPolicy._distribution", distribution)
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)
    policy = SharedGPUModelPolicy(model, None, policy_id="test", device="cpu", gpu_lock=lock)
    if fail:
        with pytest.raises(ValueError):
            policy._distribution(None)
    else:
        assert policy._distribution(None).device.type == "cpu"
    assert model.device == policy.device == "cpu"
    assert not lock.locked()


@pytest.mark.parametrize("selected", [False, True, "extra"])
def test_six_checkpoint_groups_are_started_concurrently(monkeypatch, tmp_path, selected):
    monkeypatch.setenv("TRADINGAGENTS_OHLCV_SNAPSHOT_DIR", str(tmp_path))
    (tmp_path / "training-status.json").write_text(json.dumps({"status": "completed"}))
    parts = [3, 6] if selected is True else list(range(1, 7))
    if selected == "extra":
        parts += [3, 6]
    gate = Barrier(len(parts))
    seen = []

    def part(number, *args):
        gate.wait(timeout=5)
        seen.append(number)
        return number

    monkeypatch.setattr("training.scheduler.parallel_evaluation.run_part", part)
    monkeypatch.setattr("training.scheduler.parallel_evaluation.load_tasks", lambda _: [{}] * 5)
    monkeypatch.setattr("training.scheduler.parallel_evaluation.validate_eval_tasks", lambda *a: None)
    monkeypatch.setattr("training.scheduler.parallel_evaluation.SFTTrainConfig.from_json", lambda _: SimpleNamespace(
        output_dir=str(tmp_path), train_path="train", validation_path="validation"))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("sys.argv", ["parallel", "--config", "config", "--tasks", "tasks", "--snapshot-dir", str(tmp_path)] + (["--extra-three-six"] if selected == "extra" else (["--parts-three-six"] if selected else [])))
    main()
    assert sorted(seen) == sorted(parts)
    status = json.loads((tmp_path / ("evaluation-parts-3-6-status.json" if selected is True else "evaluation-parallel-status.json")).read_text())
    assert status["status"] == "completed" and status["planned_tasks"] == len(parts) * 5
