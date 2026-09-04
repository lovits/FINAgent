from tradingagents.scheduler.trajectory import SchedulerStep, SchedulerTrajectory
from training.scheduler.audit import (
    audit_trajectory,
    parse_portfolio_rating,
    parse_trader_action,
)


def _trajectory() -> SchedulerTrajectory:
    trajectory = SchedulerTrajectory(
        "trajectory-1",
        "run-1",
        "task-1",
        "static",
        "static-v1",
        "AAPL",
        "2026-01-05",
    )
    trajectory.add_step(
        SchedulerStep(
            0,
            {"news_report": ""},
            "prompt-news",
            ["<ACT_NEWS>"],
            "<ACT_NEWS>",
            "News Analyst",
            state_after={"news_report": "evidence"},
        )
    )
    trajectory.add_step(
        SchedulerStep(
            1,
            {
                "news_report": "evidence",
                "final_trade_decision": "**Rating**: Hold",
            },
            "prompt-stop",
            ["<ACT_STOP>"],
            "<ACT_STOP>",
            None,
            state_after={"final_trade_decision": "**Rating**: Hold"},
        )
    )
    trajectory.final_outputs = {
        "investment_plan": "Hold",
        "trader_investment_plan": "**Action**: Hold\nFINAL TRANSACTION PROPOSAL: HOLD",
        "final_trade_decision": "**Rating**: Hold",
    }
    trajectory.provenance = {"selected_analysts": ["news"]}
    trajectory.execution_status = "completed"
    return trajectory


def test_parses_original_structured_output_shapes() -> None:
    assert parse_trader_action("**Action**: Sell") == "Sell"
    assert parse_trader_action("FINAL TRANSACTION PROPOSAL: **BUY**") == "Buy"
    assert parse_portfolio_rating("**Rating**: Overweight") == "Overweight"
    assert parse_portfolio_rating("no rating here") is None


def test_audit_accepts_complete_legal_trajectory() -> None:
    trajectory = _trajectory()
    audit = audit_trajectory(trajectory)
    assert audit.audit_status == "accepted"
    assert not audit.errors
    assert trajectory.audit_status == "accepted"


def test_audit_checks_only_task_selected_analyst_reports() -> None:
    trajectory = _trajectory()
    trajectory.provenance["selected_analysts"] = ["news", "fundamentals"]

    audit = audit_trajectory(trajectory)

    assert audit.audit_status == "rejected"
    assert "selected_analyst_reports_complete" in audit.errors


def test_audit_rejects_missing_stop_and_execution_failure() -> None:
    trajectory = _trajectory()
    trajectory.steps.pop()
    trajectory.execution_status = "failed"
    audit = audit_trajectory(trajectory)
    assert audit.audit_status == "rejected"
    assert "completion_valid" in audit.errors
    assert "execution_status:failed" in audit.errors
