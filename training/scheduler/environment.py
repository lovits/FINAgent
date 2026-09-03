"""Adapter that exposes TradingAgents as a grouped scheduler-rollout environment."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.scheduler.policy import SchedulerPolicy

from .rollout_runner import RolloutResult


class TradingAgentsRolloutEnvironment:
    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        selected_analysts: Sequence[str] = ("market", "social", "news", "fundamentals"),
        graph_factory: Callable[..., TradingAgentsGraph] = TradingAgentsGraph,
    ):
        self.config = dict(config)
        self.selected_analysts = tuple(selected_analysts)
        self.graph_factory = graph_factory
        self._static_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _task_values(task: Mapping[str, Any]) -> tuple[str, str, str, str]:
        ticker = str(task["ticker"])
        trade_date = str(task["trade_date"])
        asset_type = str(task.get("asset_type", "stock"))
        task_id = str(task.get("task_id", f"{ticker}:{trade_date}:{asset_type}"))
        return task_id, ticker, trade_date, asset_type

    def static_state(self, task: Mapping[str, Any]) -> dict[str, Any]:
        task_id, ticker, trade_date, asset_type = self._task_values(task)
        if task_id not in self._static_cache:
            config = dict(self.config, scheduler_mode="static")
            graph = self.graph_factory(
                self.selected_analysts,
                config=config,
                debug=False,
            )
            final_state, _ = graph.propagate(ticker, trade_date, asset_type=asset_type)
            self._static_cache[task_id] = dict(final_state)
        return deepcopy(self._static_cache[task_id])

    def run(
        self,
        task: Mapping[str, Any],
        policy: SchedulerPolicy,
        *,
        seed: int,
    ) -> RolloutResult:
        task_id, ticker, trade_date, asset_type = self._task_values(task)
        random.seed(seed)
        config = dict(
            self.config,
            scheduler_mode="learned",
            scheduler_trace_enabled=False,
        )
        graph = self.graph_factory(
            self.selected_analysts,
            config=config,
            debug=False,
            scheduler_policy=policy,
        )
        final_state, _ = graph.propagate(ticker, trade_date, asset_type=asset_type)
        recorder = graph.trajectory_recorder
        if recorder is None or recorder.current is None:
            raise RuntimeError("learned graph did not produce a scheduler trajectory")
        recorder.current.task_id = task_id
        return RolloutResult(
            trajectory=recorder.current,
            final_state=dict(final_state),
            static_state=self.static_state(task),
        )
