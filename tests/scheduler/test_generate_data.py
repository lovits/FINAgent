import json

import pytest

from training.scheduler import generate_data


class _StaticRunner:
    def __init__(self, config, *, selected_analysts):
        self.config = config
        self.selected_analysts = selected_analysts

    def run(self, task):
        return {
            "record_type": "static_langgraph",
            "status": "accepted",
            "task": dict(task),
            "provenance": {"graph_mode": "static"},
            "scheduler_examples": [],
        }


def test_load_tasks_and_generate_static_data_without_teacher_key(tmp_path, monkeypatch):
    task_path = tmp_path / "tasks.jsonl"
    task_path.write_text(
        json.dumps({"ticker": "NVDA", "trade_date": "2025-01-02"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(generate_data, "StaticLangGraphRunner", _StaticRunner)

    tasks = generate_data.load_tasks(task_path)
    counts = generate_data.generate(
        tasks=tasks,
        output_dir=tmp_path / "output",
        policy_kind="static",
        trajectories_per_task=1,
        selected_analysts=("market",),
        min_rating_similarity=0.5,
    )

    assert counts == {"accepted": 1, "rejected": 0, "skipped": 0}
    saved = json.loads(
        (tmp_path / "output" / "accepted.jsonl").read_text(encoding="utf-8")
    )
    assert saved["record_type"] == "static_langgraph"
    assert saved["provenance"]["graph_mode"] == "static"
    assert ":static:0:v1:" in saved["generation_key"]
    assert saved["generation_signature"]
    assert saved["trajectory_id"]
    assert "OPENROUTER_API_KEY" not in json.dumps(saved)

    with pytest.raises(FileExistsError, match="resume"):
        generate_data.generate(
            tasks=tasks,
            output_dir=tmp_path / "output",
            policy_kind="static",
            trajectories_per_task=1,
            selected_analysts=("market",),
            min_rating_similarity=0.5,
        )

    resumed = generate_data.generate(
        tasks=tasks,
        output_dir=tmp_path / "output",
        policy_kind="static",
        trajectories_per_task=1,
        selected_analysts=("market",),
        min_rating_similarity=0.5,
        resume=True,
    )
    assert resumed == {"accepted": 0, "rejected": 0, "skipped": 1}

    with pytest.raises(ValueError, match="configuration differs"):
        generate_data.generate(
            tasks=tasks,
            output_dir=tmp_path / "output",
            policy_kind="static",
            trajectories_per_task=1,
            selected_analysts=("market",),
            min_rating_similarity=0.5,
            config={"temperature": 0.9},
            resume=True,
        )

    generate_data.write_generation_manifest(
        output_dir=tmp_path / "output",
        task_path=task_path,
        tasks=tasks,
        policy_kind="static",
        trajectories_per_task=1,
        selected_analysts=("market",),
        resume=True,
        counts=resumed,
        generation_signature=saved["generation_signature"],
    )
    manifest = json.loads(
        (tmp_path / "output" / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["output_record_counts"] == {"accepted": 1, "rejected": 0}
    assert manifest["secrets_recorded"] is False


def test_select_generation_tasks_keeps_split_and_family_balance():
    tasks = [
        {
            "task_id": f"{split}-{family}-{index}",
            "split": split,
            "seed_family": family,
        }
        for split in ("train", "test")
        for family in ("positive_momentum", "negative_momentum")
        for index in range(3)
    ]

    selected = generate_data.select_generation_tasks(
        tasks,
        dataset_split="train",
        tasks_per_family=2,
    )
    assert len(selected) == 4
    assert {task["split"] for task in selected} == {"train"}
    assert [task["seed_family"] for task in selected].count("positive_momentum") == 2
    assert [task["seed_family"] for task in selected].count("negative_momentum") == 2
