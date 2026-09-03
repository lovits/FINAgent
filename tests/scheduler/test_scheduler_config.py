import json
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG


def test_scheduler_defaults_preserve_static_mode() -> None:
    assert DEFAULT_CONFIG["orchestration_mode"] == "static"
    assert DEFAULT_CONFIG["scheduler_max_steps"] == 16
    assert DEFAULT_CONFIG["scheduler_max_context_tokens"] == 32768
    assert DEFAULT_CONFIG["teacher_model"] == "google/gemini-3.8-flash"
    assert DEFAULT_CONFIG["scheduler_base_model"] == "Qwen/Qwen3-1.7B"


def test_scheduler_config_artifact_paths_form_one_pipeline() -> None:
    root = Path(__file__).resolve().parents[2]

    def config(name: str) -> dict:
        return json.loads((root / "configs" / "scheduler" / name).read_text())

    sft = config("sft-qwen3-1p7b.json")
    rollout = config("rollout-qwen3-1p7b.json")
    grpo = config("grpo-qwen3-1p7b.json")
    eval_sft = config("eval-sft-qwen3-1p7b.json")
    eval_grpo = config("eval-grpo-qwen3-1p7b.json")

    assert rollout["static_trajectories_path"] == (
        "data/scheduler/v1/static/train/accepted.jsonl"
    )
    assert rollout["active_adapter_path"] == f"{sft['output_dir']}/checkpoint-best"
    assert rollout["reference_adapter_path"] == f"{sft['output_dir']}/checkpoint-best"
    assert grpo["rollout_path"] == f"{rollout['output_dir']}/grpo.jsonl"
    assert grpo["sft_adapter_path"] == f"{sft['output_dir']}/checkpoint-best"
    assert eval_sft["adapter_path"] == f"{sft['output_dir']}/checkpoint-best"
    assert eval_grpo["adapter_path"] == grpo["output_dir"]
