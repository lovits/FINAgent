from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import PolicyDecision
from tradingagents.scheduler.teacher_policy import TeacherGatewayError
from training.scheduler.provider_preflight import run_preflight


def test_provider_preflight_reports_valid_action_without_secret(monkeypatch) -> None:
    monkeypatch.setattr(
        "training.scheduler.provider_preflight.TeacherSchedulerPolicy.select_action",
        lambda self, context: PolicyDecision(
            SchedulerAction.MARKET,
            policy_id=self.policy_id,
        ),
    )
    result = run_preflight("teacher-test")
    assert result == {
        "status": "passed",
        "model": "teacher-test",
        "action": "<ACT_MARKET>",
        "decision_attempts": 1,
    }


def test_provider_preflight_reports_sanitized_failure(monkeypatch) -> None:
    def fail(self, context):
        raise TeacherGatewayError("HTTP 403: provider rejected request")

    monkeypatch.setattr(
        "training.scheduler.provider_preflight.TeacherSchedulerPolicy.select_action",
        fail,
    )
    result = run_preflight("teacher-test")
    assert result["status"] == "failed"
    assert "403" in result["error"]
