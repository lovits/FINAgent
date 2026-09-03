"""Grouped environment rollouts and GRPO-row materialization."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tradingagents.scheduler.policy import SchedulerPolicy
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .advantage import group_relative_advantages
from .reward import RewardBreakdown, RewardConfig, score_trajectory


@dataclass(frozen=True)
class RolloutResult:
    trajectory: SchedulerTrajectory
    final_state: dict[str, Any]
    static_state: dict[str, Any]


@dataclass(frozen=True)
class ScoredRollout:
    result: RolloutResult
    reward: RewardBreakdown
    advantage: float


class SchedulerEnvironment(Protocol):
    def run(
        self,
        task: Mapping[str, Any],
        policy: SchedulerPolicy,
        *,
        seed: int,
    ) -> RolloutResult: ...


class GroupRolloutRunner:
    def __init__(
        self,
        environment: SchedulerEnvironment,
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
        task: Mapping[str, Any],
        policy: SchedulerPolicy,
        *,
        base_seed: int,
    ) -> list[ScoredRollout]:
        results = [
            self.environment.run(task, policy, seed=base_seed + index)
            for index in range(self.group_size)
        ]
        rewards = [
            score_trajectory(
                result.trajectory,
                result.final_state,
                result.static_state,
                config=self.reward_config,
            )
            for result in results
        ]
        advantages = group_relative_advantages([reward.total for reward in rewards])
        return [
            ScoredRollout(result, reward, advantage)
            for result, reward, advantage in zip(results, rewards, advantages, strict=True)
        ]


def grpo_rows(scored_rollouts: Iterable[ScoredRollout]) -> list[dict[str, Any]]:
    rows = []
    for rollout in scored_rollouts:
        trajectory = rollout.result.trajectory
        for step in trajectory.steps:
            if step.logprob is None:
                raise ValueError("GRPO rollout step is missing old policy logprob")
            ref_logprob = step.metadata.get("ref_logprob")
            if ref_logprob is None:
                raise ValueError("GRPO rollout step is missing reference policy logprob")
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "task_id": trajectory.task_id,
                    "step_id": step.step_id,
                    "trajectory_status": trajectory.status,
                    "serialized_state": step.serialized_state,
                    "valid_actions": step.valid_actions,
                    "selected_action": step.selected_action,
                    "advantage": rollout.advantage,
                    "reward": rollout.reward.total,
                    "reward_components": rollout.reward.to_dict(),
                    "old_logprob": step.logprob,
                    "ref_logprob": float(ref_logprob),
                }
            )
    return rows


def write_grpo_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
