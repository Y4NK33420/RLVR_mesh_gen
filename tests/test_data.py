from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from r4.data import normalize_vertices, sample_paths, sort_vertices_and_remap_faces, write_manifest


def test_normalize_vertices_clips_to_target_range() -> None:
    vertices = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, -5.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )

    normalized = normalize_vertices(vertices, target_abs_range=1.0)

    assert normalized.shape == vertices.shape
    assert float(np.abs(normalized).max()) <= 1.0 + 1e-8


def test_sort_vertices_remaps_faces_consistently() -> None:
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    sorted_vertices, remapped_faces, order = sort_vertices_and_remap_faces(vertices, faces)

    assert sorted_vertices.shape == vertices.shape
    assert remapped_faces.shape == faces.shape
    assert sorted(order.tolist()) == [0, 1, 2]


def test_manifest_write_and_sample(tmp_path: Path) -> None:
    all_paths = [tmp_path / f"mesh_{idx}.obj" for idx in range(20)]
    sampled = sample_paths(all_paths, limit=5, seed=123)

    assert len(sampled) == 5

    output = tmp_path / "manifest.json"
    write_manifest(sampled, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["num_samples"] == 5
    assert len(payload["paths"]) == 5
