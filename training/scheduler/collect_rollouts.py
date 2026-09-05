"""Collect one on-policy GRPO trajectory group per selected Train task."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from tradingagents.scheduler.hf_policy import load_shared_hf_scheduler_policies
from tradingagents.scheduler.store import TrajectoryStore, write_json_atomic
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .auto_review import AutomaticReviewer, ReviewUnavailable
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
    reward_mode: str = "static_agreement"
    judge_model: str = "z-ai/glm-5.3-flash"
    expert_model: str = "z-ai/glm-5.3-flash"
    parallel_workers: int = 8
    trajectory_workers: int = 4
    resume: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> RolloutConfig:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**value)


def collect(config: RolloutConfig) -> dict[str, int]:
    if config.reward_mode not in {"automatic", "static_agreement"}:
        raise ValueError("unknown reward_mode")
    tasks = load_tasks(config.tasks_path, split="train")
    if config.limit is not None:
        tasks = tasks[: config.limit]
    if not tasks:
        raise ValueError("rollout task set is empty")
    if config.parallel_workers < 1 or config.trajectory_workers < 1:
        raise ValueError("parallel worker counts must be positive")
    static_values = (TrajectoryStore(config.static_trajectories_path).load()
                     if config.reward_mode == "static_agreement" else [])
    static_by_key = {
        (trajectory.task_id, trajectory.data_snapshot_id): trajectory
        for trajectory in static_values
    }
    runtime_config = scheduler_runtime_config(
        {
            "orchestration_mode": "learned",
            "llm_provider": "openrouter",
            "quick_think_llm": config.expert_model,
            "deep_think_llm": config.expert_model,
            "temperature": 0.0,
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
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
    output = Path(config.output_dir)
    raw_store = TrajectoryStore(output / "raw-trajectories.jsonl")
    existing_raw = raw_store.load()
    if existing_raw and not config.resume:
        raise FileExistsError("rollout output exists; enable resume to reuse it")
    existing_by_task = {}
    for trajectory in existing_raw:
        existing_by_task.setdefault(trajectory.task_id, []).append(trajectory)
    review_path = output / "quality_reviews.json"
    existing_reviews = (json.loads(review_path.read_text()) if review_path.exists() else {})
    policy_lock = Lock()
    raw_lock = Lock()

    def append_raw(trajectory):
        with raw_lock:
            if not raw_store.contains(trajectory.trajectory_id):
                raw_store.append(trajectory)

    def run_group(index, task):
        task_config, selected_analysts = resolve_task_runtime(task, runtime_config)
        key = (task["task_id"], task.get("data_snapshot_id"))
        if config.reward_mode == "static_agreement" and key not in static_by_key:
            raise ValueError(f"missing Static reference for {key}")
        static_reference = static_by_key.get(key)
        if static_reference is not None:
            _validate_static_reference(static_reference, task, selected_analysts)
        known = existing_by_task.get(task["task_id"], [])
        if len(known) > config.group_size:
            raise ValueError(f"too many existing trajectories for {task['task_id']}")
        reviewer = (AutomaticReviewer(
            config.judge_model,
            existing_reviews={key: value for key, value in existing_reviews.items()
                              if key in {item.trajectory_id for item in known}},
        ) if config.reward_mode == "automatic" else None)
        runner = GroupRolloutRunner(
            TradingAgentsSchedulerEnvironment(
                task_config,
                selected_analysts=selected_analysts,
            ),
            group_size=config.group_size,
            reward_config=reward_config,
            reward_scorer=reviewer,
            trajectory_sink=append_raw,
            policy_lock=policy_lock,
            trajectory_workers=config.trajectory_workers,
        )
        try:
            scored = runner.run_task(
                task, active_policy=active, reference_policy=reference,
                static_reference=static_reference, run_id=config.run_id,
                base_seed=config.seed + index * config.group_size,
                existing_trajectories=known,
            )
        except ReviewUnavailable:
            return index, [], [], reviewer.snapshot(), {
                "task_id": task["task_id"], "reason": "review_or_environment_unavailable"
            }
        if reviewer is not None and all(value.advantage == 0 for value in scored):
            return index, [], [], reviewer.snapshot(), {
                "task_id": task["task_id"], "reason": "zero_group_reward_variance"
            }
        if any(not value.trajectory.steps for value in scored):
            return index, [], [], reviewer.snapshot() if reviewer else {}, {
                "task_id": task["task_id"], "reason": "empty_trajectory"
            }
        return index, [value.trajectory for value in scored], grpo_rows(scored), \
            reviewer.snapshot() if reviewer else {}, None

    results = {}
    with ThreadPoolExecutor(max_workers=min(len(tasks), config.parallel_workers)) as pool:
        futures = {pool.submit(run_group, index, task): task["task_id"]
                   for index, task in enumerate(tasks)}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(f"rollout worker failed for {task_id}: {exc}") from exc
            results[result[0]] = result
            reviews = dict(existing_reviews)
            skipped = []
            for _, _, _, task_reviews, skip in (results[index] for index in sorted(results)):
                reviews.update(task_reviews)
                if skip is not None:
                    skipped.append(skip)
            if config.reward_mode == "automatic":
                write_json_atomic(review_path, reviews)
                write_json_atomic(output / "skipped_groups.json", {"groups": skipped})

    trajectories = []
    rows = []
    reviews = dict(existing_reviews)
    skipped_groups = []
    for index in sorted(results):
        _, group_trajectories, group_rows, group_reviews, skip = results[index]
        trajectories.extend(group_trajectories)
        rows.extend(group_rows)
        reviews.update(group_reviews)
        if skip is not None:
            skipped_groups.append(skip)

    trajectory_store = TrajectoryStore(output / "trajectories.jsonl")
    for trajectory in trajectories:
        trajectory_store.append(trajectory)
    write_grpo_rows(rows, output / "grpo.jsonl")
    counts = {"tasks": len(tasks), "trajectories": len(trajectories), "rows": len(rows)}
    if config.reward_mode == "automatic":
        counts["skipped_groups"] = len(skipped_groups)
        write_json_atomic(review_path, reviews)
        write_json_atomic(output / "skipped_groups.json", {"groups": skipped_groups})
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
