import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import PolicyDecision, SchedulerContext
from tradingagents.scheduler.recorder import TrajectoryRecorder
from tradingagents.scheduler.trajectory import ExecutionCost, SchedulerTrajectory


def _state() -> dict:
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2026-01-05",
        "asset_type": "stock",
        "news_report": "",
        "investment_debate_state": {"history": "", "count": 0},
        "risk_debate_state": {"history": "", "count": 0},
    }


def _recorder() -> TrajectoryRecorder:
    return TrajectoryRecorder(
        SchedulerTrajectory(
            "trajectory-1",
            "run-1",
            "task-1",
            "learned",
            "local-v1",
            "AAPL",
            "2026-01-05",
        )
    )


def _context(state: dict, action: SchedulerAction) -> SchedulerContext:
    return SchedulerContext(
        "task-1",
        state,
        "prompt",
        (action,),
        ("news",),
    )


def test_recorder_pairs_decision_with_expert_observation() -> None:
    recorder = _recorder()
    state = _state()
    recorder.record_decision(
        _context(state, SchedulerAction.NEWS),
        PolicyDecision(SchedulerAction.NEWS, "local-v1", logprob=-0.4),
    )
    state["news_report"] = "complete"
    recorder.record_observation(
        state,
        observation_ref="node-1",
        cost=ExecutionCost(agent_calls=1, tool_calls=2, input_tokens=100),
    )
    trajectory = recorder.finalize(state)

    assert trajectory.steps[0].state_before["news_report"] == ""
    assert trajectory.steps[0].state_after["news_report"] == "complete"
    assert trajectory.steps[0].old_logprob == -0.4
    assert trajectory.cost_total.tool_calls == 2


def test_stop_is_recorded_without_pending_observation() -> None:
    recorder = _recorder()
    state = _state()
    state["final_trade_decision"] = "**Rating**: Hold"
    recorder.record_decision(
        _context(state, SchedulerAction.STOP),
        PolicyDecision(SchedulerAction.STOP, "local-v1"),
    )
    trajectory = recorder.finalize(state)
    assert trajectory.steps[0].agent_node is None


def test_recorder_rejects_overlapping_pending_decisions() -> None:
    recorder = _recorder()
    context = _context(_state(), SchedulerAction.NEWS)
    decision = PolicyDecision(SchedulerAction.NEWS, "local-v1")
    recorder.record_decision(context, decision)
    with pytest.raises(ValueError, match="previous scheduler decision"):
        recorder.record_decision(context, decision)
    with pytest.raises(ValueError, match="pending Expert"):
        recorder.finalize(_state())
