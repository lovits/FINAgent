"""Generate same-task trajectory groups and materialize GRPO action rows."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tradingagents.scheduler.contracts import ActionLogprobPolicy, SchedulerPolicy
from tradingagents.scheduler.policy import ReferenceScoredPolicy
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .advantage import group_relative_advantages
from .audit import audit_trajectory
from .environment import EnvironmentRunResult
from .reward import RewardBreakdown, RewardConfig, score_trajectory

GRPO_SCHEMA_VERSION = "scheduler-grpo-v1"


class RolloutEnvironment(Protocol):
    def run(
        self,
        task: dict[str, Any],
        *,
        mode: str,
        run_id: str,
        policy: SchedulerPolicy,
        trajectory_id: str | None = None,
    ) -> EnvironmentRunResult: ...


@dataclass(frozen=True)
class ScoredRollout:
    trajectory: SchedulerTrajectory
    reward: RewardBreakdown
    advantage: float


class GroupRolloutRunner:
    def __init__(
        self,
        environment: RolloutEnvironment,
        *,
        group_size: int = 4,
        reward_config: RewardConfig | None = None,
    ):
        if group_size < 2:
            raise ValueError("group_size must be at least two")
        self.environment = environment
        self.group_size = group_size
        self.reward_config = reward_config or RewardConfig()

    def run_task(
        self,
        task: dict[str, Any],
        *,
        active_policy: SchedulerPolicy,
        reference_policy: ActionLogprobPolicy,
        static_reference: SchedulerTrajectory,
        run_id: str,
        base_seed: int,
    ) -> list[ScoredRollout]:
        policy = ReferenceScoredPolicy(active_policy, reference_policy)
        trajectories = []
        for index in range(self.group_size):
            _set_seed(base_seed + index)
            result = self.environment.run(
                task,
                mode="learned",
                run_id=run_id,
                policy=policy,
                trajectory_id=f"{run_id}:{task['task_id']}:{index}",
            )
            audit_trajectory(result.trajectory)
            trajectories.append(result.trajectory)
        rewards = [
            score_trajectory(value, static_reference, config=self.reward_config)
            for value in trajectories
        ]
        advantages = group_relative_advantages([value.total for value in rewards])
        scored = []
        for trajectory, reward, advantage in zip(
            trajectories, rewards, advantages, strict=True
        ):
            trajectory.reward = reward.to_dict()
            scored.append(ScoredRollout(trajectory, reward, advantage))
        return scored


def grpo_rows(scored_rollouts: Iterable[ScoredRollout]) -> list[dict[str, Any]]:
    rows = []
    for rollout in scored_rollouts:
        group_id = f"{rollout.trajectory.run_id}:{rollout.trajectory.task_id}"
        for step in rollout.trajectory.steps:
            if step.old_logprob is None or step.ref_logprob is None:
                raise ValueError("GRPO scheduler step requires old and reference logprobs")
            rows.append(
                {
                    "schema_version": GRPO_SCHEMA_VERSION,
                    "rollout_group_id": group_id,
                    "trajectory_id": rollout.trajectory.trajectory_id,
                    "task_id": rollout.trajectory.task_id,
                    "step_id": step.step_id,
                    "serialized_state": step.serialized_state,
                    "valid_actions": step.valid_actions,
                    "selected_action": step.selected_action,
                    "old_logprob": step.old_logprob,
                    "ref_logprob": step.ref_logprob,
                    "reward_total": rollout.reward.total,
                    "reward_components": rollout.reward.to_dict(),
                    "advantage": rollout.advantage,
                }
            )
    return rows


def write_grpo_rows(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
