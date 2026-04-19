from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import trimesh


@dataclass(frozen=True)
class PreprocessConfig:
    target_abs_range: float = 1.0
    sort_vertices_zyx: bool = True
    sort_face_indices: bool = False


def normalize_vertices(vertices: np.ndarray, target_abs_range: float = 1.0) -> np.ndarray:
    vertices_arr = np.asarray(vertices, dtype=np.float64)
    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3]")

    min_corner = vertices_arr.min(axis=0)
    max_corner = vertices_arr.max(axis=0)
    center = (min_corner + max_corner) / 2.0

    centered = vertices_arr - center
    max_abs = np.abs(centered).max()
    if max_abs < 1e-12:
        return centered

    return centered * (target_abs_range / max_abs)


def sort_vertices_and_remap_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int64)

    if vertices_arr.ndim != 2 or vertices_arr.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3]")
    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
        raise ValueError("faces must have shape [M, 3]")

    order = np.lexsort((vertices_arr[:, 0], vertices_arr[:, 1], vertices_arr[:, 2]))
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)

    sorted_vertices = vertices_arr[order]
    remapped_faces = inverse_order[faces_arr]

    return sorted_vertices, remapped_faces, order


def preprocess_mesh(vertices: np.ndarray, faces: np.ndarray, config: PreprocessConfig | None = None) -> dict[str, np.ndarray]:
    cfg = config or PreprocessConfig()

    processed_vertices = normalize_vertices(vertices, target_abs_range=cfg.target_abs_range)
    processed_faces = np.asarray(faces, dtype=np.int64)

    order = np.arange(processed_vertices.shape[0], dtype=np.int64)
    if cfg.sort_vertices_zyx:
        processed_vertices, processed_faces, order = sort_vertices_and_remap_faces(
            processed_vertices,
            processed_faces,
        )

    if cfg.sort_face_indices:
        # Sorting face indices changes winding order; keep disabled unless tokenizer requires it.
        processed_faces = np.sort(processed_faces, axis=1)

    return {
        "vertices": processed_vertices,
        "faces": processed_faces,
        "vertex_order": order,
    }


def load_mesh_arrays(mesh_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh_path_obj = Path(mesh_path)
    loaded = trimesh.load(mesh_path_obj, force="mesh", process=False)

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry found in scene: {mesh_path_obj}")
        merged = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        return np.asarray(merged.vertices, dtype=np.float64), np.asarray(merged.faces, dtype=np.int64)

    return np.asarray(loaded.vertices, dtype=np.float64), np.asarray(loaded.faces, dtype=np.int64)


def find_shapenet_meshes(
    dataset_root: str | Path,
    synset_id: str = "03001627",
    extensions: tuple[str, ...] = (".obj", ".glb", ".ply", ".off"),
) -> list[Path]:
    root = Path(dataset_root)
    synset_dir = root / synset_id
    if not synset_dir.exists():
        return []

    exts = {ext.lower() for ext in extensions}
    mesh_paths = [
        p for p in synset_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    ]

    return sorted(mesh_paths)


def sample_paths(paths: list[Path], limit: int | None, seed: int = 42) -> list[Path]:
    if limit is None or limit >= len(paths):
        return paths

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=limit, replace=False)
    chosen = [paths[int(i)] for i in indices]

    return sorted(chosen)


def write_manifest(paths: list[Path], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "num_samples": len(paths),
        "paths": [str(p.as_posix()) for p in paths],
    }

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
