from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass(frozen=True)
class RewardConfig:
    boolean_bonus: float = 1.0
    invalid_penalty_scale: float = 0.25
    weight_ber: float = 0.5
    weight_triangle_quality: float = 0.5
    run_self_intersection_check: bool = True


def build_trimesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int64)

    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3]")
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("faces must have shape [M, 3]")

    return trimesh.Trimesh(vertices=vertices_arr, faces=faces_arr, process=False, validate=False)


def boundary_edge_ratio(faces: np.ndarray) -> float:
    faces_arr = np.asarray(faces, dtype=np.int64)
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("faces must have shape [M, 3]")

    if faces_arr.size == 0:
        return 1.0

    e01 = faces_arr[:, [0, 1]]
    e12 = faces_arr[:, [1, 2]]
    e20 = faces_arr[:, [2, 0]]
    edges = np.concatenate([e01, e12, e20], axis=0)
    edges = np.sort(edges, axis=1)

    _, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = np.count_nonzero(counts == 1)
    total = counts.size

    return float(boundary / total) if total > 0 else 1.0


def triangle_quality_score(vertices: np.ndarray, faces: np.ndarray) -> float:
    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int64)

    if faces_arr.size == 0:
        return 0.0

    tris = vertices_arr[faces_arr]
    a = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
    b = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
    c = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)

    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    denom = (a * a) + (b * b) + (c * c) + 1e-12

    quality = (4.0 * np.sqrt(3.0) * area) / denom
    quality = np.clip(quality, 0.0, 1.0)

    return float(np.mean(quality))


def _self_intersection_status(vertices: np.ndarray, faces: np.ndarray, enabled: bool) -> tuple[bool, bool]:
    if not enabled:
        return False, False

    try:
        import open3d as o3d
    except ImportError:
        return False, False

    vertices_f32 = np.asarray(vertices, dtype=np.float32)
    faces_i32 = np.asarray(faces, dtype=np.int32)

    mesh = o3d.t.geometry.TriangleMesh(
        vertex_positions=o3d.core.Tensor(vertices_f32),
        triangle_indices=o3d.core.Tensor(faces_i32),
    )

    value = mesh.is_self_intersecting()
    if hasattr(value, "item"):
        return bool(value.item()), True
    return bool(value), True


def evaluate_mesh_reward(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: RewardConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RewardConfig()

    mesh = build_trimesh(vertices=vertices, faces=faces)
    watertight = bool(mesh.is_watertight)

    self_intersection, was_checked = _self_intersection_status(
        vertices=mesh.vertices,
        faces=mesh.faces,
        enabled=cfg.run_self_intersection_check,
    )

    ber = boundary_edge_ratio(mesh.faces)
    tri_quality = triangle_quality_score(mesh.vertices, mesh.faces)

    continuous = (cfg.weight_triangle_quality * tri_quality) + (cfg.weight_ber * (1.0 - ber))

    boolean_pass = watertight and (not self_intersection)
    reward = continuous + cfg.boolean_bonus if boolean_pass else continuous * cfg.invalid_penalty_scale
    cost = 0.0 if boolean_pass else 1.0

    return {
        "reward": float(reward),
        "cost": float(cost),
        "continuous_reward": float(continuous),
        "boolean_pass": bool(boolean_pass),
        "watertight": bool(watertight),
        "self_intersection": bool(self_intersection),
        "self_intersection_checked": bool(was_checked),
        "boundary_edge_ratio": float(ber),
        "triangle_quality_score": float(tri_quality),
        "num_vertices": int(mesh.vertices.shape[0]),
        "num_faces": int(mesh.faces.shape[0]),
    }
