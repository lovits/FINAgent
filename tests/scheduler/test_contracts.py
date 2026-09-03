import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import PolicyDecision, SchedulerContext
from tradingagents.scheduler.policy import CallableSchedulerPolicy


def test_scheduler_context_normalizes_actions() -> None:
    context = SchedulerContext(
        task_id="AAPL_2026-01-05",
        state={},
        serialized_state="prompt",
        valid_actions=("<ACT_MARKET>",),  # type: ignore[arg-type]
        selected_analysts=("market",),
        history=("<ACT_NEWS>",),  # type: ignore[arg-type]
    )

    assert context.valid_actions == (SchedulerAction.MARKET,)
    assert context.history == (SchedulerAction.NEWS,)


def test_scheduler_context_requires_valid_budget_and_actions() -> None:
    with pytest.raises(ValueError, match="valid action"):
        SchedulerContext("task", {}, "prompt", (), ("market",))
    with pytest.raises(ValueError, match="step budget"):
        SchedulerContext(
            "task",
            {},
            "prompt",
            (SchedulerAction.MARKET,),
            ("market",),
            step=17,
            max_steps=16,
        )


def test_policy_decision_validates_correction_metadata() -> None:
    corrected = PolicyDecision(
        SchedulerAction.NEWS,
        policy_id="teacher-v1",
        decision_attempts=2,
        correction_succeeded=True,
    )
    assert corrected.action is SchedulerAction.NEWS

    with pytest.raises(ValueError, match="requires two"):
        PolicyDecision(
            SchedulerAction.NEWS,
            policy_id="teacher-v1",
            correction_succeeded=True,
        )


def test_policy_decision_rejects_non_finite_logprob() -> None:
    with pytest.raises(ValueError, match="finite"):
        PolicyDecision(SchedulerAction.NEWS, policy_id="local-v1", logprob=float("nan"))


def test_callable_policy_rejects_masked_action() -> None:
    context = SchedulerContext(
        "task",
        {},
        "prompt",
        (SchedulerAction.MARKET,),
        ("market",),
    )
    policy = CallableSchedulerPolicy(lambda _: SchedulerAction.NEWS)
    with pytest.raises(ValueError, match="masked action"):
        policy.select_action(context)
