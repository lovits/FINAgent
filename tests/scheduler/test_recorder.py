from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import PolicyDecision, SchedulerContext
from tradingagents.scheduler.recorder import TrajectoryRecorder
from tradingagents.scheduler.trajectory_store import TrajectoryStore


def _context(step, valid):
    return SchedulerContext(
        state={},
        serialized_state=f'{{"step":{step}}}',
        valid_actions=tuple(valid),
        selected_analysts=("market",),
        step=step,
    )


def test_recorder_persists_versioned_teacher_trajectory(tmp_path):
    path = tmp_path / "trajectories.jsonl"
    stats = {"tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
    recorder = TrajectoryRecorder(path, stats_provider=lambda: dict(stats))
    recorder.begin(
        task_id="task",
        ticker="NVDA",
        trade_date="2025-01-02",
        asset_type="stock",
        policy_id="teacher-v1",
        metadata={"source": "strong_teacher_verified"},
    )
    recorder.record_decision(
        _context(0, [SchedulerAction.MARKET]),
        PolicyDecision(
            SchedulerAction.MARKET,
            logprob=-0.2,
            metadata={
                "teacher_model": "google/gemini-3.8-flash",
                "teacher_prompt_version": "v1",
            },
        ),
    )
    stats.update({"tool_calls": 2, "input_tokens": 100, "output_tokens": 25})
    recorder.record_decision(
        _context(1, [SchedulerAction.STOP]),
        PolicyDecision(SchedulerAction.STOP),
    )
    completed = recorder.finalize(
        {
            "trader_investment_plan": "Buy",
            "final_trade_decision": "Hold",
        },
        status="accepted",
        verifier_version="v1",
    )

    assert completed.steps[0].observation_summary == '{"step":1}'
    assert completed.steps[0].tool_calls == 2
    assert completed.steps[0].input_tokens == 100
    assert completed.steps[0].output_tokens == 25
    assert completed.steps[0].latency_ms >= 0
    assert completed.teacher_model == "google/gemini-3.8-flash"
    assert completed.teacher_prompt_version == "v1"
    saved = list(TrajectoryStore(path).load())
    assert saved[0].trajectory_id == completed.trajectory_id
    assert saved[0].status == "accepted"
