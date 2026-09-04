import pytest

from training.scheduler.profile import (
    resolve_task_runtime,
    validate_shallow_runtime,
    validate_training_profile,
)


def test_training_profile_accepts_one_or_more_input_analysts() -> None:
    assert validate_training_profile(("market",)) == ("market",)
    assert validate_training_profile(("market", "news")) == ("market", "news")


def test_training_profile_rejects_empty_analyst_pool() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_training_profile(())


def test_task_input_is_the_source_of_analysts_language_and_depth() -> None:
    config, analysts = resolve_task_runtime(
        {
            "selected_analysts": ["market", "news"],
            "research_depth": "shallow",
            "output_language": "Chinese",
        },
        {
            "output_language": "English",
            "max_debate_rounds": 3,
            "max_risk_discuss_rounds": 3,
        },
    )

    assert analysts == ("market", "news")
    assert config["output_language"] == "Chinese"
    assert config["max_debate_rounds"] == 1
    assert config["max_risk_discuss_rounds"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("selected_analysts", [], "at least one"),
        ("research_depth", "medium", "research_depth=shallow"),
        ("output_language", "English", "output_language=Chinese"),
    ),
)
def test_task_input_rejects_unsupported_profile(field, value, message) -> None:
    task = {
        "selected_analysts": ["market"],
        "research_depth": "shallow",
        "output_language": "Chinese",
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        resolve_task_runtime(task, {})


def test_training_profile_rejects_medium_or_deep_runtime() -> None:
    validate_shallow_runtime(
        {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    )

    with pytest.raises(ValueError, match="requires shallow"):
        validate_shallow_runtime(
            {"max_debate_rounds": 3, "max_risk_discuss_rounds": 3}
        )
