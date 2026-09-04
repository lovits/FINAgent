import json
from types import SimpleNamespace

from training.scheduler.provenance import (
    code_provenance,
    expert_config_hash,
    generation_config_hash,
    trajectory_provenance,
)


def _config(**overrides):
    return {
        "llm_provider": "openrouter",
        "quick_think_llm": "expert-quick",
        "deep_think_llm": "expert-deep",
        "teacher_model": "z-ai/glm-5.3-flash",
        "scheduler_max_steps": 16,
        **overrides,
    }


def test_generation_hash_is_stable_and_excludes_credentials() -> None:
    first = generation_config_hash(
        _config(api_key="secret-one", OPENROUTER_API_KEY="secret-two"),
        mode="teacher",
        policy_id="teacher-v1",
        selected_analysts=("market", "news"),
    )
    second = generation_config_hash(
        _config(api_key="different", OPENROUTER_API_KEY="also-different"),
        mode="teacher",
        policy_id="teacher-v1",
        selected_analysts=("market", "news"),
    )
    changed_model = generation_config_hash(
        _config(teacher_model="another-model"),
        mode="teacher",
        policy_id="teacher-v1",
        selected_analysts=("market", "news"),
    )

    assert first == second
    assert first != changed_model
    assert len(first) == 64


def test_expert_hash_changes_only_with_execution_configuration() -> None:
    first = expert_config_hash(
        _config(OPENROUTER_API_KEY="secret-one"),
        selected_analysts=("market", "news"),
    )
    same = expert_config_hash(
        _config(OPENROUTER_API_KEY="secret-two", teacher_model="different-teacher"),
        selected_analysts=("market", "news"),
    )
    different = expert_config_hash(
        _config(quick_think_llm="different-expert"),
        selected_analysts=("market", "news"),
    )

    assert first == same
    assert first != different


def test_expert_hash_normalizes_environment_numeric_strings() -> None:
    from_environment = expert_config_hash(
        _config(temperature="0.0", max_debate_rounds="1"),
        selected_analysts=("market", "news"),
    )
    programmatic = expert_config_hash(
        _config(temperature=0.0, max_debate_rounds=1),
        selected_analysts=("market", "news"),
    )

    assert from_environment == programmatic


def test_trajectory_provenance_records_versions_without_secrets() -> None:
    provenance = trajectory_provenance(
        config=_config(OPENROUTER_API_KEY="must-not-leak"),
        task={
            "dataset_version": "scheduler-tasks-v2",
            "split": "train",
            "seed_family": "earnings_window",
            "sector": "Technology",
            "information_cutoff": "2026-08-27T23:59:59Z",
            "data_snapshot_id": "snapshot-1",
            "memory_snapshot_id": "memory-1",
            "research_depth": "shallow",
            "output_language": "Chinese",
        },
        mode="teacher",
        policy_id="teacher-v1",
        selected_analysts=("market", "news"),
        code={"git_commit": "abc123", "git_dirty": False},
    )

    assert provenance["teacher_model"] == "z-ai/glm-5.3-flash"
    assert provenance["task_dataset_version"] == "scheduler-tasks-v2"
    assert provenance["data_snapshot_id"] == "snapshot-1"
    assert provenance["selected_analysts"] == ["market", "news"]
    assert provenance["research_depth"] == "shallow"
    assert provenance["output_language"] == "Chinese"
    assert provenance["action_schema_version"] == "scheduler-actions-v1"
    assert provenance["scheduler_prompt_version"] == "scheduler-prompt-v7"
    assert provenance["orchestration_profile_version"] == (
        "multi-analyst-shallow-v1"
    )
    assert len(provenance["expert_config_hash"]) == 64
    assert provenance["trajectory_schema_version"] == "scheduler-trajectory-v1"
    assert "must-not-leak" not in json.dumps(provenance)


def test_code_provenance_degrades_to_unknown_outside_git(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "training.scheduler.provenance.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert code_provenance(tmp_path) == {"git_commit": "unknown", "git_dirty": False}


def test_code_provenance_records_commit_and_dirty_flag(monkeypatch, tmp_path) -> None:
    outputs = iter(("abc123\n", " M tracked.py\n"))
    monkeypatch.setattr(
        "training.scheduler.provenance.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=next(outputs)),
    )

    assert code_provenance(tmp_path) == {"git_commit": "abc123", "git_dirty": True}
