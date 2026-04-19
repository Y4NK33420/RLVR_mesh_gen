from __future__ import annotations

import numpy as np
import trimesh

from r4.reward_engine import RewardConfig, boundary_edge_ratio, evaluate_mesh_reward, triangle_quality_score


def test_boundary_edge_ratio_closed_box_is_zero() -> None:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    ber = boundary_edge_ratio(box.faces)

    assert np.isclose(ber, 0.0)


def test_open_mesh_has_positive_cost_and_fails_boolean() -> None:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    open_mesh = box.copy()
    open_mesh.faces = open_mesh.faces[:-1]

    cfg = RewardConfig(run_self_intersection_check=False)
    result = evaluate_mesh_reward(open_mesh.vertices, open_mesh.faces, cfg)

    assert result["boolean_pass"] is False
    assert result["cost"] == 1.0


def test_triangle_quality_bounds() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    score = triangle_quality_score(vertices, faces)

    assert 0.0 <= score <= 1.0
