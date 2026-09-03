"""Collect grouped scheduler rollouts with active and frozen-reference policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.scheduler.hf_policy import load_hf_scheduler_policy
from tradingagents.scheduler.policy import ReferenceScoredPolicy
from tradingagents.scheduler.trajectory_store import TrajectoryStore

from .environment import TradingAgentsRolloutEnvironment
from .generate_data import load_tasks
from .rollout_runner import GroupRolloutRunner, grpo_rows, write_grpo_jsonl


@dataclass(frozen=True)
class RolloutConfig:
    base_model: str
    active_adapter_path: str
    reference_adapter_path: str
    tasks_path: str
    output_dir: str
    base_model_revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    group_size: int = 4
    action_temperature: float = 1.0
    base_seed: int = 42
    selected_analysts: tuple[str, ...] = ("market", "social", "news", "fundamentals")

    @classmethod
    def from_json(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "selected_analysts" in payload:
            payload["selected_analysts"] = tuple(payload["selected_analysts"])
        return cls(**payload)


def collect(config: RolloutConfig) -> dict[str, int]:
    common = {
        "scheduler_mode": "learned",
        "scheduler_base_model": config.base_model,
        "scheduler_base_model_revision": config.base_model_revision,
        "scheduler_tokenizer_revision": config.tokenizer_revision,
        "scheduler_dtype": config.dtype,
        "scheduler_device": config.device,
    }
    active = load_hf_scheduler_policy(
        {
            **common,
            "scheduler_adapter_path": config.active_adapter_path,
            "scheduler_action_temperature": config.action_temperature,
        }
    )
    reference = load_hf_scheduler_policy(
        {
            **common,
            "scheduler_adapter_path": config.reference_adapter_path,
            "scheduler_action_temperature": 0.0,
        }
    )
    policy = ReferenceScoredPolicy(active, reference)
    runtime_config = dict(DEFAULT_CONFIG, **common, scheduler_trace_enabled=False)
    environment = TradingAgentsRolloutEnvironment(
        runtime_config,
        selected_analysts=config.selected_analysts,
    )
    runner = GroupRolloutRunner(environment, group_size=config.group_size)
    output = Path(config.output_dir)
    trajectory_store = TrajectoryStore(output / "trajectories.jsonl")
    all_rows = []
    task_count = 0
    trajectory_count = 0
    for task_index, task in enumerate(load_tasks(config.tasks_path)):
        scored = runner.run_task(
            task,
            policy,
            base_seed=config.base_seed + task_index * config.group_size,
        )
        for rollout in scored:
            trajectory = rollout.result.trajectory
            trajectory.reward = rollout.reward.to_dict()
            trajectory_store.append(trajectory)
            trajectory_count += 1
        all_rows.extend(grpo_rows(scored))
        task_count += 1
    write_grpo_jsonl(all_rows, output / "grpo_rows.jsonl")
    return {
        "tasks": task_count,
        "trajectories": trajectory_count,
        "training_rows": len(all_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(RolloutConfig.from_json(args.config)), sort_keys=True))


if __name__ == "__main__":
    main()
