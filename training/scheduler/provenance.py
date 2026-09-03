"""Reproducibility metadata for scheduler trajectories and generation runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tradingagents.scheduler.actions import ACTION_SCHEMA_VERSION
from tradingagents.scheduler.prompt import (
    PROMPT_VERSION,
    STATE_SCHEMA_VERSION,
    TEACHER_PROMPT_VERSION,
)
from tradingagents.scheduler.registry import AGENT_CATALOG_VERSION
from tradingagents.scheduler.trajectory import TRAJECTORY_SCHEMA_VERSION

_CONFIG_FINGERPRINT_FIELDS = (
    "llm_provider",
    "quick_think_llm",
    "deep_think_llm",
    "backend_url",
    "output_language",
    "max_debate_rounds",
    "max_risk_discuss_rounds",
    "scheduler_max_steps",
    "scheduler_max_context_tokens",
    "teacher_provider",
    "teacher_model",
    "teacher_temperature",
    "teacher_seed",
    "data_vendors",
    "tool_vendors",
)


def code_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    commit = _git(root, "rev-parse", "HEAD") or "unknown"
    dirty = bool(_git(root, "status", "--porcelain", "--untracked-files=no"))
    return {"git_commit": commit, "git_dirty": dirty}


def generation_config_hash(
    config: Mapping[str, Any],
    *,
    mode: str,
    policy_id: str,
    selected_analysts: Sequence[str],
) -> str:
    payload = {
        "mode": mode,
        "policy_id": policy_id,
        "selected_analysts": list(selected_analysts),
        "config": {
            key: config.get(key)
            for key in _CONFIG_FINGERPRINT_FIELDS
            if key in config
        },
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def trajectory_provenance(
    *,
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    mode: str,
    policy_id: str,
    selected_analysts: Sequence[str],
    code: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(code),
        "task_dataset_version": task.get("dataset_version"),
        "task_split": task.get("split"),
        "seed_family": task.get("seed_family"),
        "sector": task.get("sector"),
        "information_cutoff": task.get("information_cutoff"),
        "data_snapshot_id": task.get("data_snapshot_id"),
        "memory_snapshot_id": task.get("memory_snapshot_id"),
        "expert_provider": config.get("llm_provider"),
        "quick_model": config.get("quick_think_llm"),
        "deep_model": config.get("deep_think_llm"),
        "teacher_model": config.get("teacher_model") if mode == "teacher" else None,
        "teacher_prompt_version": (
            TEACHER_PROMPT_VERSION if mode == "teacher" else None
        ),
        "policy_id": policy_id,
        "selected_analysts": list(selected_analysts),
        "agent_catalog_version": AGENT_CATALOG_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "scheduler_prompt_version": PROMPT_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "generation_config_hash": generation_config_hash(
            config,
            mode=mode,
            policy_id=policy_id,
            selected_analysts=selected_analysts,
        ),
    }


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()
