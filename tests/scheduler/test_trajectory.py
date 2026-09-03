import json

import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.replay import replay_decisions
from tradingagents.scheduler.trajectory import SchedulerTrajectory, TrajectoryStep
from tradingagents.scheduler.trajectory_store import TrajectoryStore


def _trajectory():
    trajectory = SchedulerTrajectory(
        trajectory_id="traj-1",
        task_id="NVDA-2025-01-02",
        run_id="run-1",
        mode="learned",
        policy_id="mock-v1",
        ticker="NVDA",
        trade_date="2025-01-02",
    )
    trajectory.add_step(
        TrajectoryStep(
            step_id=0,
            serialized_state='{"step":0}',
            valid_actions=[SchedulerAction.MARKET.value, SchedulerAction.NEWS.value],
            selected_action=SchedulerAction.MARKET.value,
            policy_id="mock-v1",
            agent_node="Market Analyst",
        )
    )
    return trajectory


def test_trajectory_round_trip_and_replay(tmp_path):
    store = TrajectoryStore(tmp_path / "trajectories.jsonl")
    store.append(_trajectory())

    loaded = list(store.load())
    assert len(loaded) == 1
    assert loaded[0].to_dict() == _trajectory().to_dict()
    replay = list(replay_decisions(loaded[0]))
    assert replay[0].recorded_action is SchedulerAction.MARKET


def test_trajectory_rejects_non_contiguous_steps_and_invalid_action():
    trajectory = _trajectory()
    with pytest.raises(ValueError, match="expected step_id 1"):
        trajectory.add_step(
            TrajectoryStep(
                step_id=2,
                serialized_state="{}",
                valid_actions=[SchedulerAction.NEWS.value],
                selected_action=SchedulerAction.NEWS.value,
                policy_id="mock-v1",
            )
        )

    with pytest.raises(ValueError, match="is not valid"):
        TrajectoryStep(
            step_id=0,
            serialized_state="{}",
            valid_actions=[SchedulerAction.NEWS.value],
            selected_action=SchedulerAction.TRADER.value,
            policy_id="mock-v1",
        )


def test_store_reports_corrupt_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"schema_version": "v0"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported trajectory schema"):
        list(TrajectoryStore(path).load())
