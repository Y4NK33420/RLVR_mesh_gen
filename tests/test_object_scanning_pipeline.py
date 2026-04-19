from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from r4.object_scanning_pipeline import (
    ObjectScanningPipelineConfig,
    _split_count_triplet,
    run_object_scanning_pipeline,
    stratified_split_records,
)


def test_split_count_triplet_sum_and_nonnegative() -> None:
    train_count, val_count, test_count = _split_count_triplet(11, 0.7, 0.15, 0.15)

    assert train_count + val_count + test_count == 11
    assert train_count >= 0
    assert val_count >= 0
    assert test_count >= 0


def test_stratified_split_preserves_all_records() -> None:
    records = []
    for idx in range(12):
        records.append({"sample_id": f"a_{idx}", "category": "A"})
    for idx in range(7):
        records.append({"sample_id": f"b_{idx}", "category": "B"})

    splits = stratified_split_records(records, 0.7, 0.15, 0.15, seed=123)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])

    assert total == len(records)


def test_pipeline_runs_on_tiny_fixture_dataset(tmp_path: Path) -> None:
    input_root = tmp_path / "Object Scanning"
    output_root = tmp_path / "artifacts" / "gso_preprocessed"

    categories = ["Furniture", "Electronics", "Toys"]
    for category in categories:
        for idx in range(2):
            obj_dir = input_root / category / "Sub" / f"Item_{idx}"
            obj_dir.mkdir(parents=True, exist_ok=True)

            box = trimesh.creation.box(extents=(1.0 + idx, 1.0, 1.0))
            box.export(obj_dir / f"mesh_{idx}.obj")

    cfg = ObjectScanningPipelineConfig(
        input_root=input_root,
        output_root=output_root,
        min_vertices=4,
        min_faces=4,
        include_self_intersection_check=False,
    )

    result = run_object_scanning_pipeline(cfg)

    assert result["summary"]["num_discovered"] == 6
    assert result["summary"]["num_processed"] == 6
    assert (output_root / "manifests" / "train.json").exists()
    assert (output_root.parent / "gso_preprocessed.zip").exists()
