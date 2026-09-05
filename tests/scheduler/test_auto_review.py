import json
from copy import deepcopy

import pytest

from tests.scheduler.test_generate import _complete_trajectory
from tradingagents.scheduler.trajectory import ExecutionCost
from training.scheduler.auto_review import (
    AutomaticReviewer,
    ReviewUnavailable,
    automatic_reward,
    review_documents,
    review_payload,
    validate_assessment,
)


def assessment(score=3):
    return {
        name: {
            "score": score,
            "reason": "supported by the supplied report",
            "citations": [{"document": "market_report", "quote": "market"}],
        }
        for name in ("evidence_alignment", "logical_consistency", "risk_disclosure")
    }


class Response:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self.value)}}]}


def test_review_payload_hides_policy_mode_cost_and_baseline():
    trajectory = _complete_trajectory("hidden-id", "teacher", "hidden-run")
    payload = review_payload(trajectory)
    assert set(payload) == {"task", "documents"}
    assert "teacher" not in json.dumps(payload)
    assert "hidden-id" not in json.dumps(payload)
    assert "cost_total" not in payload


def test_unverifiable_evidence_is_retried_once_then_not_used_as_reward():
    trajectory = _complete_trajectory("t", "teacher", "r")
    invalid = assessment()
    invalid["risk_disclosure"]["citations"][0]["quote"] = "invented quote"
    calls = []

    def transport(*args, **kwargs):
        calls.append(kwargs)
        return Response(invalid)

    reviewer = AutomaticReviewer(transport=transport, environ={"OPENROUTER_API_KEY": "test"})
    review = reviewer.assess(trajectory)
    assert review["status"] == "unavailable"
    assert len(calls) == 2
    with pytest.raises(ReviewUnavailable):
        automatic_reward(trajectory, review)


def test_corrected_judge_response_is_accepted():
    responses = iter([Response({}), Response(assessment())])
    reviewer = AutomaticReviewer(
        transport=lambda *a, **kw: next(responses), environ={"OPENROUTER_API_KEY": "test"}
    )
    review = reviewer.assess(_complete_trajectory("t", "teacher", "r"))
    assert review["status"] == "reviewed"
    assert review["attempts"] == 2
    assert review["quality"] == 0.75


def test_existing_review_is_reused_without_another_judge_request():
    trajectory = _complete_trajectory("cached", "teacher", "r")
    cached = {"status": "reviewed", "quality": 0.75, "dimensions": assessment()}
    reviewer = AutomaticReviewer(
        transport=lambda *a, **kw: pytest.fail("cached review must not call judge"),
        existing_reviews={trajectory.trajectory_id: cached},
    )
    assert reviewer.assess(trajectory) == cached
    assert reviewer.snapshot() == {trajectory.trajectory_id: cached}


@pytest.mark.parametrize("value", [True, -1, 5, float("nan"), "3"])
def test_invalid_scores_are_rejected(value):
    answer = assessment()
    answer["evidence_alignment"]["score"] = value
    with pytest.raises(ValueError):
        validate_assessment(answer, review_documents(_complete_trajectory("t", "static", "r")))


def test_rule_failure_never_calls_judge_and_receives_negative_reward():
    trajectory = _complete_trajectory("t", "teacher", "r")
    trajectory.execution_status = "budget_exhausted"
    reviewer = AutomaticReviewer(transport=lambda *a, **kw: pytest.fail("unexpected API"))
    review = reviewer.assess(trajectory)
    assert automatic_reward(trajectory, review).total == -1


def test_network_failure_is_unscorable_not_a_policy_penalty():
    trajectory = _complete_trajectory("t", "teacher", "r")
    trajectory.execution_status = "failed"
    trajectory.failure_reason = "NoMarketDataError: upstream not available"
    assert AutomaticReviewer(environ={}).assess(trajectory)["status"] == "unavailable"


def test_good_quality_dominates_cheap_bad_report_and_cost_is_bounded():
    trajectory = _complete_trajectory("t", "teacher", "r")
    bad = automatic_reward(trajectory, {"status": "reviewed", "quality": 0.25})
    trajectory.cost_total = ExecutionCost(agent_calls=100, input_tokens=1_000_000)
    good = automatic_reward(trajectory, {"status": "reviewed", "quality": 0.75})
    assert good.total > bad.total
    assert good.cost_penalty == 0.15


def test_different_static_decision_does_not_change_automatic_reward():
    trajectory = _complete_trajectory("t", "teacher", "r")
    reviewer = AutomaticReviewer(
        transport=lambda *a, **kw: Response(assessment()), environ={"OPENROUTER_API_KEY": "test"}
    )
    other = deepcopy(trajectory)
    first = reviewer(trajectory, other)
    other.final_outputs["final_trade_decision"] = "**Rating**: Sell"
    assert reviewer(trajectory, other) == first
