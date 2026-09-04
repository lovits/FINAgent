"""Collect one on-policy GRPO trajectory group per selected Train task."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

from tradingagents.scheduler.hf_policy import load_shared_hf_scheduler_policies
from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .environment import TradingAgentsSchedulerEnvironment
from .generate import load_tasks
from .profile import resolve_task_runtime, validate_shallow_runtime
from .reward import RewardConfig
from .rollout import GroupRolloutRunner, grpo_rows, write_grpo_rows
from .runtime_config import scheduler_runtime_config


@dataclass(frozen=True)
class RolloutConfig:
    tasks_path: str
    static_trajectories_path: str
    active_adapter_path: str
    reference_adapter_path: str
    output_dir: str
    run_id: str
    reward_config_path: str | None = "configs/scheduler/reward-v1.json"
    base_model: str = "Qwen/Qwen3-1.7B"
    base_revision: str | None = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    group_size: int = 4
    action_temperature: float = 0.8
    max_context_tokens: int = 32768
    max_steps: int = 16
    limit: int | None = None
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> RolloutConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**value)


def collect(config: RolloutConfig) -> dict[str, int]:
    tasks = load_tasks(config.tasks_path, split="train")
    if config.limit is not None:
        tasks = tasks[: config.limit]
    static_values = TrajectoryStore(config.static_trajectories_path).load()
    static_by_key = {
        (trajectory.task_id, trajectory.data_snapshot_id): trajectory
        for trajectory in static_values
    }
    runtime_config = scheduler_runtime_config(
        {
            "orchestration_mode": "learned",
            "scheduler_base_model": config.base_model,
            "scheduler_base_revision": config.base_revision,
            "scheduler_max_context_tokens": config.max_context_tokens,
            "scheduler_max_steps": config.max_steps,
            "scheduler_action_temperature": config.action_temperature,
        }
    )
    validate_shallow_runtime(runtime_config)
    active, reference = load_shared_hf_scheduler_policies(
        runtime_config,
        active_adapter_path=config.active_adapter_path,
        reference_adapter_path=config.reference_adapter_path,
        temperature=config.action_temperature,
    )
    reward_config = RewardConfig()
    if config.reward_config_path:
        reward_config = RewardConfig(
            **json.loads(Path(config.reward_config_path).read_text(encoding="utf-8"))
        )
    trajectories = []
    rows = []
    for index, task in enumerate(tasks):
        task_config, selected_analysts = resolve_task_runtime(task, runtime_config)
        key = (task["task_id"], task.get("data_snapshot_id"))
        if key not in static_by_key:
            raise ValueError(f"missing Static reference for {key}")
        static_reference = static_by_key[key]
        _validate_static_reference(static_reference, task, selected_analysts)
        runner = GroupRolloutRunner(
            TradingAgentsSchedulerEnvironment(
                task_config,
                selected_analysts=selected_analysts,
            ),
            group_size=config.group_size,
            reward_config=reward_config,
        )
        scored = runner.run_task(
            task,
            active_policy=active,
            reference_policy=reference,
            static_reference=static_reference,
            run_id=config.run_id,
            base_seed=config.seed + index * config.group_size,
        )
        trajectories.extend(value.trajectory for value in scored)
        rows.extend(grpo_rows(scored))

    output = Path(config.output_dir)
    trajectory_store = TrajectoryStore(output / "trajectories.jsonl")
    for trajectory in trajectories:
        trajectory_store.append(trajectory)
    write_grpo_rows(rows, output / "grpo.jsonl")
    counts = {"tasks": len(tasks), "trajectories": len(trajectories), "rows": len(rows)}
    write_json_atomic(
        output / "rollout_manifest.json",
        {"config": asdict(config), "counts": counts},
    )
    return counts


def _validate_static_reference(
    trajectory: SchedulerTrajectory,
    task: dict,
    selected_analysts: tuple[str, ...],
) -> None:
    expected = {
        "selected_analysts": list(selected_analysts),
        "research_depth": task["research_depth"],
        "output_language": task["output_language"],
    }
    for field, value in expected.items():
        if trajectory.provenance.get(field) != value:
            raise ValueError(f"Static reference does not match task input: {field}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(RolloutConfig.from_json(args.config)), sort_keys=True))


if __name__ == "__main__":
    main()
