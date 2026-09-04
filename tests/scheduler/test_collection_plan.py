from training.scheduler.collection_plan import (
    build_collection_tasks,
    write_collection_tasks,
)


def _task(index: int, split: str = "reserve") -> dict:
    return {
        "task_id": f"TASK_{index}",
        "ticker": f"T{index}",
        "trade_date": "2026-01-05",
        "split": split,
        "seed_family": "quiet_control" if index % 2 else "volume_shock",
        "selected_analysts": ["market"] if index % 3 else ["market", "news"],
    }


def test_builds_promoted_balanced_extension_plan() -> None:
    tasks = [_task(index) for index in range(8)] + [_task(99, "validation")]

    selected = build_collection_tasks(
        tasks,
        paired_count=2,
        teacher_only_count=3,
        source_split="reserve",
        offset=1,
    )

    assert len(selected) == 5
    assert [task["collection_role"] for task in selected] == [
        "paired",
        "paired",
        "teacher_only",
        "teacher_only",
        "teacher_only",
    ]
    assert {task["split"] for task in selected} == {"train"}
    assert {task["source_split"] for task in selected} == {"reserve"}
    assert "TASK_99" not in {task["task_id"] for task in selected}


def test_writes_plan_manifest_with_trajectory_count(tmp_path) -> None:
    tasks = build_collection_tasks(
        [_task(index) for index in range(5)],
        paired_count=2,
        teacher_only_count=3,
    )

    manifest = write_collection_tasks(tasks, tmp_path / "tasks.jsonl")

    assert manifest["task_count"] == 5
    assert manifest["planned_trajectory_count"] == 7
    assert manifest["role_counts"] == {"paired": 2, "teacher_only": 3}
    assert (tmp_path / "tasks.manifest.json").exists()
