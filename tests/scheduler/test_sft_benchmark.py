import json
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import pytest

from training.scheduler.sft_benchmark import limit_expert_requests, validate_tasks, wait_for_training


def test_seeds_require_three_seen_and_four_unseen():
    tasks = [{"task_id": str(i), "evaluation_bucket": "seen_sft" if i < 3 else "unseen_sft",
              "sft_part3_samples_seen": int(i < 3), "sft_part6_samples_seen": int(i < 3)} for i in range(7)]
    validate_tasks(tasks)
    tasks[6]["sft_part6_samples_seen"] = 1
    with pytest.raises(ValueError, match="overlaps"):
        validate_tasks(tasks)


def test_api_gate_bounds_concurrency_and_restores_method(monkeypatch):
    from langchain_openai import ChatOpenAI

    lock = Lock()
    active = peak = 0

    def call(self):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1

    monkeypatch.setattr(ChatOpenAI, "_generate", call)
    with limit_expert_requests(2), ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: ChatOpenAI._generate(None), range(8)))
    assert peak == 2 and ChatOpenAI._generate is call


def test_gpu_gate_waits_for_rl_without_allocating_during_training(monkeypatch, tmp_path):
    status = tmp_path / "rl.json"
    status.write_text(json.dumps({"status": "running", "round": 2, "stage": "rollout"}))
    def available():
        assert json.loads(status.read_text())["status"] == "completed"
        return True
    monkeypatch.setattr("torch.cuda.is_available", available)
    monkeypatch.setattr("torch.cuda.mem_get_info", lambda: (20 * 1024**3, 24 * 1024**3))
    monkeypatch.setattr("training.scheduler.sft_benchmark.time.sleep", lambda _: status.write_text(json.dumps({"status": "completed"})))
    wait_for_training(status, tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "waiting_for_gpu"
