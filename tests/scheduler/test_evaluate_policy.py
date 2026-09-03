import json

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.trajectory import SchedulerTrajectory, TrajectoryStep
from training.scheduler.evaluate_policy import (
    evaluate_trajectory,
    summarize,
    write_evaluation_report,
)


def _trajectory():
    trajectory = SchedulerTrajectory(
        trajectory_id="trajectory",
        task_id="task",
        run_id="run",
        mode="learned",
        policy_id="sft",
        ticker="NVDA",
        trade_date="2025-01-02",
    )
    trajectory.add_step(
        TrajectoryStep(
            step_id=0,
            serialized_state="{}",
            valid_actions=[SchedulerAction.TRADER.value],
            selected_action=SchedulerAction.TRADER.value,
            policy_id="sft",
            agent_node="Trader",
            tool_calls=2,
            input_tokens=100,
            output_tokens=50,
            latency_ms=20,
        )
    )
    return trajectory


def test_evaluation_record_and_report_are_mode_comparable(tmp_path):
    state = {
        "investment_plan": "plan",
        "trader_investment_plan": "FINAL TRANSACTION PROPOSAL: **BUY**",
        "final_trade_decision": "**Rating**: Overweight",
    }
    static = evaluate_trajectory(_trajectory(), state, mode_label="static")
    learned = evaluate_trajectory(_trajectory(), state, mode_label="learned-sft")
    summary = summarize([static, learned])

    assert summary["static"]["completion_rate"] == 1.0
    assert summary["learned-sft"]["mean_tool_calls"] == 2
    path = tmp_path / "report.json"
    write_evaluation_report([static, learned], path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert set(report["summary"]) == {"static", "learned-sft"}
