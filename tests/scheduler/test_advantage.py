import math

import pytest

from training.scheduler.advantage import group_relative_advantages


def test_group_advantages_are_centered_and_ordered():
    advantages = group_relative_advantages([0.8, 0.4, 0.1, -1.0])

    assert math.isclose(sum(advantages), 0.0, abs_tol=1e-7)
    assert advantages[0] > advantages[1] > advantages[2] > advantages[3]


def test_zero_variance_group_skips_update_with_zero_advantage():
    assert group_relative_advantages([0.5, 0.5, 0.5, 0.5]) == [0.0] * 4


def test_group_advantage_requires_at_least_two_rollouts():
    with pytest.raises(ValueError, match="at least two"):
        group_relative_advantages([1.0])
