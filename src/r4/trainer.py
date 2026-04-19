from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .grpo import group_relative_advantages
from .lagrangian import PIConstraintController
from .reward_engine import RewardConfig, evaluate_mesh_reward


@dataclass(frozen=True)
class TrainConfig:
    group_size: int = 8
    target_cost: float = 0.05
    lambda_init: float = 0.01
    controller_kp: float = 0.1
    controller_ki: float = 0.02
    advantage_epsilon: float = 1e-8


def lagrangian_group_objective(
    rewards: np.ndarray,
    costs: np.ndarray,
    lambda_value: float,
    target_cost: float,
) -> float:
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    costs_arr = np.asarray(costs, dtype=np.float64)

    if rewards_arr.shape != costs_arr.shape:
        raise ValueError("rewards and costs must have the same shape")

    return float(np.mean(rewards_arr - lambda_value * (costs_arr - target_cost)))


class OfflineGRPOTrainer:
    """Offline scaffold: computes rewards, advantages, and PI-updated Lagrangian metrics."""

    def __init__(self, reward_config: RewardConfig | None = None, train_config: TrainConfig | None = None):
        self.reward_config = reward_config or RewardConfig()
        self.train_config = train_config or TrainConfig()

        self.controller = PIConstraintController(
            lambda_value=self.train_config.lambda_init,
            kp=self.train_config.controller_kp,
            ki=self.train_config.controller_ki,
            target_cost=self.train_config.target_cost,
            min_lambda=0.0,
            max_lambda=None,
        )
        self.global_step = 0

    def evaluate_group(self, meshes: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
        if len(meshes) == 0:
            raise ValueError("meshes cannot be empty")

        per_mesh = [
            evaluate_mesh_reward(vertices, faces, config=self.reward_config)
            for vertices, faces in meshes
        ]

        rewards = np.asarray([item["reward"] for item in per_mesh], dtype=np.float64)
        costs = np.asarray([item["cost"] for item in per_mesh], dtype=np.float64)

        advantages = group_relative_advantages(rewards, epsilon=self.train_config.advantage_epsilon)

        mean_cost = float(np.mean(costs))
        lambda_before = float(self.controller.lambda_value)
        lambda_after = float(self.controller.update(observed_cost=mean_cost))

        objective = lagrangian_group_objective(
            rewards=rewards,
            costs=costs,
            lambda_value=lambda_after,
            target_cost=self.train_config.target_cost,
        )

        return {
            "group_size": int(len(meshes)),
            "rewards": rewards,
            "costs": costs,
            "advantages": advantages,
            "mean_reward": float(rewards.mean()),
            "mean_cost": mean_cost,
            "lambda_before": lambda_before,
            "lambda_after": lambda_after,
            "lagrangian_objective": objective,
            "per_mesh": per_mesh,
        }

    def should_resample_group(self, rewards: np.ndarray) -> bool:
        rewards_arr = np.asarray(rewards, dtype=np.float64)
        if rewards_arr.size == 0:
            return True

        return bool(rewards_arr.std() < self.train_config.advantage_epsilon)

    def step(self, meshes: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
        result = self.evaluate_group(meshes)
        self.global_step += 1
        result["step"] = int(self.global_step)
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": int(self.global_step),
            "train_config": asdict(self.train_config),
            "reward_config": asdict(self.reward_config),
            "controller": {
                "lambda_value": float(self.controller.lambda_value),
                "kp": float(self.controller.kp),
                "ki": float(self.controller.ki),
                "target_cost": float(self.controller.target_cost),
                "min_lambda": float(self.controller.min_lambda),
                "max_lambda": (None if self.controller.max_lambda is None else float(self.controller.max_lambda)),
                "integral_error": float(self.controller._integral_error),
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.global_step = int(state.get("global_step", 0))

        controller_state = state.get("controller", {})
        self.controller.lambda_value = float(controller_state.get("lambda_value", self.controller.lambda_value))
        self.controller.kp = float(controller_state.get("kp", self.controller.kp))
        self.controller.ki = float(controller_state.get("ki", self.controller.ki))
        self.controller.target_cost = float(controller_state.get("target_cost", self.controller.target_cost))
        self.controller.min_lambda = float(controller_state.get("min_lambda", self.controller.min_lambda))

        max_lambda = controller_state.get("max_lambda", self.controller.max_lambda)
        self.controller.max_lambda = None if max_lambda is None else float(max_lambda)

        self.controller._integral_error = float(controller_state.get("integral_error", 0.0))
