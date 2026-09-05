import json
from types import SimpleNamespace

from training.scheduler.repair_evaluation import select_inference_device


def test_short_cpu_inference_does_not_touch_training_gpu(monkeypatch, tmp_path):
    (tmp_path / "training-status.json").write_text(json.dumps({"status": "running"}))
    monkeypatch.setattr("training.scheduler.repair_evaluation.scheduler_input_ids", lambda *a: [1] * 10)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (_ for _ in ()).throw(AssertionError("GPU touched")))
    policy = SimpleNamespace(device="cpu", tokenizer=None, max_context_tokens=32768)
    select_inference_device(policy, SimpleNamespace(serialized_state="state"), tmp_path)
    assert policy.device == "cpu"


def test_long_context_waits_for_training_completion_then_moves_to_gpu(monkeypatch, tmp_path):
    status = tmp_path / "training-status.json"
    status.write_text(json.dumps({"status": "running"}))
    monkeypatch.setattr("training.scheduler.repair_evaluation.scheduler_input_ids", lambda *a: [1] * 9000)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (20 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr("training.scheduler.repair_evaluation.time.sleep",
                        lambda _: status.write_text(json.dumps({"status": "completed"})))
    moves = []
    policy = SimpleNamespace(device="cpu", tokenizer=None, max_context_tokens=32768,
                             model=SimpleNamespace(to=lambda device: moves.append(device)))
    select_inference_device(policy, SimpleNamespace(serialized_state="state"), tmp_path)
    assert moves == ["cuda"] and policy.device == "cuda"


def test_overlength_input_is_left_to_policy_validation(monkeypatch, tmp_path):
    monkeypatch.setattr("training.scheduler.repair_evaluation.scheduler_input_ids", lambda *a: [1] * 20)
    policy = SimpleNamespace(device="cpu", tokenizer=None, max_context_tokens=10)
    select_inference_device(policy, SimpleNamespace(serialized_state="state"), tmp_path)
    assert not (tmp_path / "evaluation-repair-device.json").exists()
