import json

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import SchedulerContext
from tradingagents.scheduler.teacher_context import load_teacher_context
from tradingagents.scheduler.teacher_gateway import OpenRouterTeacherGateway
from tradingagents.scheduler.teacher_policy import StrongTeacherPolicy


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "next_agent": "<ACT_MARKET>",
                                "reason_code": "MISSING_MARKET_EVIDENCE",
                                "reason": "Need market evidence.",
                            }
                        )
                    }
                }
            ]
        }


def test_default_teacher_context_matches_runtime_registry():
    context = load_teacher_context()

    assert context["agent_catalog"]["version"] == "v1"
    assert context["state_schema"]["version"] == "v1"
    assert context["orchestration_rules"]["version"] == "v1"
    assert context["positive_examples"]
    assert context["failure_examples"]


def test_strong_teacher_policy_builds_context_and_uses_mock_gateway():
    captured = {}

    def transport(url, **kwargs):
        captured.update(kwargs["json"])
        return _Response()

    gateway = OpenRouterTeacherGateway(
        transport=transport,
        environ={"OPENROUTER_API_KEY": "test-only-key"},
    )
    policy = StrongTeacherPolicy(gateway)
    decision = policy.select_action(
        SchedulerContext(
            state={},
            serialized_state='{"reports":{}}',
            valid_actions=(SchedulerAction.MARKET,),
            selected_analysts=("market",),
            step=0,
        )
    )

    assert decision.action is SchedulerAction.MARKET
    serialized_messages = json.dumps(captured["messages"])
    assert "CORRECTED_FAILURE_EXAMPLES" in serialized_messages
    assert "test-only-key" not in serialized_messages
