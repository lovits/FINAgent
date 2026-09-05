"""Generate same-task trajectory groups and materialize GRPO action rows."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from tradingagents.scheduler.contracts import (
    ActionLogprobPolicy, PolicyDecision, SchedulerContext, SchedulerPolicy,
)
from tradingagents.scheduler.policy import ReferenceScoredPolicy
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .advantage import group_relative_advantages
from .audit import audit_trajectory
from .environment import EnvironmentRunResult
from .reward import RewardConfig, score_trajectory

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
    reward: Any
    advantage: float


class SeededLockedReferencePolicy:
    """Atomically sample active and reference adapters with per-trajectory RNG."""

    def __init__(self, active, reference, lock, seed):
        self.active = active
        self.reference = reference
        self.lock = lock
        self.seed = seed
        self.step = 0
        self.policy_id = active.policy_id

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        import torch

        with self.lock, torch.random.fork_rng():
            torch.manual_seed(self.seed + self.step)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed + self.step)
            self.step += 1
            decision = self.active.select_action(context)
            reference_logprobs = self.reference.action_logprobs(context)
        if decision.action not in reference_logprobs:
            raise ValueError("reference policy did not score the selected action")
        return PolicyDecision(
            decision.action, policy_id=decision.policy_id, logprob=decision.logprob,
            decision_attempts=decision.decision_attempts,
            correction_succeeded=decision.correction_succeeded,
            metadata={**decision.metadata, "ref_logprob": reference_logprobs[decision.action]},
        )


class GroupRolloutRunner:
    def __init__(
        self,
        environment: RolloutEnvironment,
        *,
        group_size: int = 4,
        reward_config: RewardConfig | None = None,
        reward_scorer: Callable | None = None,
        trajectory_sink: Callable | None = None,
        policy_lock=None,
        trajectory_workers: int = 1,
    ):
        if group_size < 2:
            raise ValueError("group_size must be at least two")
        self.environment = environment
        self.group_size = group_size
        self.reward_config = reward_config or RewardConfig()
        self.reward_scorer = reward_scorer
        self.trajectory_sink = trajectory_sink
        self.policy_lock = policy_lock
        self.trajectory_workers = trajectory_workers
        if trajectory_workers < 1:
            raise ValueError("trajectory_workers must be positive")

    def run_task(
        self,
        task: dict[str, Any],
        *,
        active_policy: SchedulerPolicy,
        reference_policy: ActionLogprobPolicy,
        static_reference: SchedulerTrajectory | None,
        run_id: str,
        base_seed: int,
        existing_trajectories: Iterable[SchedulerTrajectory] = (),
    ) -> list[ScoredRollout]:
        if static_reference is None and self.reward_scorer is None:
            raise ValueError("Static-agreement reward requires a Static reference")
        existing = {value.trajectory_id: value for value in existing_trajectories}

        def generate(index):
            identifier = f"{run_id}:{task['task_id']}:{index}"
            if identifier in existing:
                return existing[identifier]
            policy = (SeededLockedReferencePolicy(
                active_policy, reference_policy, self.policy_lock, base_seed + index
            ) if self.policy_lock is not None else ReferenceScoredPolicy(
                active_policy, reference_policy
            ))
            result = self.environment.run(
                task,
                mode="learned",
                run_id=run_id,
                policy=policy,
                trajectory_id=identifier,
            )
            audit_trajectory(result.trajectory)
            return result.trajectory

        with ThreadPoolExecutor(max_workers=min(self.group_size, self.trajectory_workers)) as pool:
            trajectories = list(pool.map(generate, range(self.group_size)))
        if self.trajectory_sink is not None:
            for trajectory in trajectories:
                if trajectory.trajectory_id not in existing:
                    self.trajectory_sink(trajectory)

        def score(value):
            return (self.reward_scorer(value, static_reference)
                    if self.reward_scorer is not None
                    else score_trajectory(value, static_reference, config=self.reward_config))

        with ThreadPoolExecutor(max_workers=min(self.group_size, self.trajectory_workers)) as pool:
            rewards = list(pool.map(score, trajectories))
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
