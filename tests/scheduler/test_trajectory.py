import pytest

from tradingagents.scheduler.trajectory import (
    ExecutionCost,
    NodeExecution,
    SchedulerStep,
    SchedulerTrajectory,
)


def _trajectory() -> SchedulerTrajectory:
    return SchedulerTrajectory(
        trajectory_id="trajectory-1",
        run_id="run-1",
        task_id="task-1",
        mode="teacher",
        policy_id="teacher-v1",
        ticker="AAPL",
        trade_date="2026-01-05",
    )


def _step(step_id: int = 0) -> SchedulerStep:
    return SchedulerStep(
        step_id=step_id,
        state_before={},
        serialized_state="prompt",
        valid_actions=["<ACT_NEWS>"],
        selected_action="<ACT_NEWS>",
        agent_node="News Analyst",
        state_after={"news_report": "done"},
        cost=ExecutionCost(agent_calls=1, input_tokens=10, output_tokens=5),
    )


def test_scheduler_step_rejects_action_node_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        SchedulerStep(
            0,
            {},
            "prompt",
            ["<ACT_NEWS>"],
            "<ACT_NEWS>",
            "Trader",
        )


def test_trajectory_requires_contiguous_scheduler_and_node_steps() -> None:
    trajectory = _trajectory()
    trajectory.add_step(_step())
    with pytest.raises(ValueError, match="expected scheduler step 1"):
        trajectory.add_step(_step(2))

    execution = NodeExecution(0, "News Analyst", "expert", {}, {}, {})
    trajectory.add_node_execution(execution)
    with pytest.raises(ValueError, match="expected node step 1"):
        trajectory.add_node_execution(NodeExecution(2, "Trader", "expert", {}, {}, {}))


def test_trajectory_round_trip_preserves_nested_types() -> None:
    trajectory = _trajectory()
    trajectory.add_step(_step())
    trajectory.add_node_execution(
        NodeExecution(
            0,
            "News Analyst",
            "expert",
            {},
            {"news_report": "done"},
            {"news_report": "done"},
            cost=ExecutionCost(agent_calls=1, latency_ms=25.0),
        )
    )

    restored = SchedulerTrajectory.from_dict(trajectory.to_dict())
    assert restored.steps[0].cost.input_tokens == 10
    assert restored.node_executions[0].cost.latency_ms == 25.0


def test_execution_cost_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="negative"):
        ExecutionCost(tool_calls=-1)
