import pytest

from training.scheduler.advantage import group_relative_advantages


def test_group_advantages_are_centered_and_ordered() -> None:
    values = group_relative_advantages([1.0, 2.0, 3.0, 4.0])
    assert sum(values) == pytest.approx(0.0)
    assert values == sorted(values)


def test_equal_rewards_produce_zero_update_signal() -> None:
    assert group_relative_advantages([0.5, 0.5, 0.5, 0.5]) == [0.0] * 4


def test_group_requires_multiple_finite_rewards() -> None:
    with pytest.raises(ValueError, match="at least two"):
        group_relative_advantages([1.0])
    with pytest.raises(ValueError, match="finite"):
        group_relative_advantages([1.0, float("nan")])
