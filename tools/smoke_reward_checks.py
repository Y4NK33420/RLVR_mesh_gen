from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r4.grpo import group_relative_advantages
from r4.lagrangian import PIConstraintController
from r4.reward_engine import RewardConfig, evaluate_mesh_reward


def print_result(title: str, result: dict[str, float | bool | int]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key in [
        "reward",
        "cost",
        "continuous_reward",
        "boolean_pass",
        "watertight",
        "self_intersection",
        "self_intersection_checked",
        "boundary_edge_ratio",
        "triangle_quality_score",
        "num_vertices",
        "num_faces",
    ]:
        print(f"{key}: {result[key]}")


def main() -> None:
    cfg = RewardConfig(
        boolean_bonus=1.0,
        invalid_penalty_scale=0.2,
        weight_ber=0.5,
        weight_triangle_quality=0.5,
        run_self_intersection_check=True,
    )

    closed = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    closed_result = evaluate_mesh_reward(closed.vertices, closed.faces, cfg)
    print_result("Closed box mesh", closed_result)

    open_mesh = closed.copy()
    open_mesh.faces = open_mesh.faces[:-1]
    open_result = evaluate_mesh_reward(open_mesh.vertices, open_mesh.faces, cfg)
    print_result("Open box mesh (one face removed)", open_result)

    sample_rewards = np.array([0.2, 0.9, 0.7, 0.2, 0.6, 1.1, 0.5, 0.4], dtype=np.float64)
    advantages = group_relative_advantages(sample_rewards)
    print("\nGRPO advantage demo")
    print("-------------------")
    print("rewards:", sample_rewards.tolist())
    print("advantages:", advantages.round(4).tolist())

    controller = PIConstraintController(lambda_value=0.01, kp=0.1, ki=0.02, target_cost=0.05)
    observed_costs = [0.4, 0.25, 0.1, 0.04, 0.03, 0.07, 0.02]

    print("\nLagrangian PI update demo")
    print("-------------------------")
    for idx, cost in enumerate(observed_costs, start=1):
        new_lambda = controller.update(observed_cost=cost)
        print(f"step {idx}: observed_cost={cost:.3f}, lambda={new_lambda:.5f}")


if __name__ == "__main__":
    main()
