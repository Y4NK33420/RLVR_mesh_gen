from __future__ import annotations

import numpy as np


def group_relative_advantages(rewards: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Compute GRPO-style normalized advantages for one reward group."""
    rewards_arr = np.asarray(rewards, dtype=np.float64)

    if rewards_arr.ndim != 1:
        raise ValueError("rewards must be a 1D array-like")
    if rewards_arr.size == 0:
        raise ValueError("rewards cannot be empty")

    group_std = rewards_arr.std()
    if group_std < epsilon:
        return np.zeros_like(rewards_arr, dtype=np.float64)

    return (rewards_arr - rewards_arr.mean()) / (group_std + epsilon)
