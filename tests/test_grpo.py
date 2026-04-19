from __future__ import annotations

import numpy as np

from r4.grpo import group_relative_advantages


def test_zero_variance_rewards_produce_zero_advantages() -> None:
    rewards = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    advantages = group_relative_advantages(rewards)

    assert np.allclose(advantages, 0.0)


def test_advantages_are_centered_for_nonconstant_rewards() -> None:
    rewards = np.array([0.1, 0.3, 0.5, 0.9], dtype=np.float64)
    advantages = group_relative_advantages(rewards)

    assert np.isclose(float(advantages.mean()), 0.0, atol=1e-8)
    assert advantages.shape == rewards.shape
