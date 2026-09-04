import pytest

from training.scheduler.profile import validate_shallow_runtime, validate_training_profile


def test_training_profile_accepts_multi_analyst_candidates() -> None:
    assert validate_training_profile(("market", "news")) == ("market", "news")


def test_training_profile_rejects_single_analyst_data() -> None:
    with pytest.raises(ValueError, match="requires at least two"):
        validate_training_profile(("market",))


def test_training_profile_rejects_medium_or_deep_runtime() -> None:
    validate_shallow_runtime(
        {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    )

    with pytest.raises(ValueError, match="requires shallow"):
        validate_shallow_runtime(
            {"max_debate_rounds": 3, "max_risk_discuss_rounds": 3}
        )
