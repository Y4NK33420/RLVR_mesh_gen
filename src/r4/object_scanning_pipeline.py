from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import zipfile

import numpy as np
import trimesh

from .data import PreprocessConfig, load_mesh_arrays, preprocess_mesh
from .reward_engine import RewardConfig, evaluate_mesh_reward


@dataclass(frozen=True)
class ObjectScanningPipelineConfig:
    input_root: Path
    output_root: Path
    mesh_extensions: tuple[str, ...] = (".obj",)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    min_vertices: int = 16
    min_faces: int = 16
    include_self_intersection_check: bool = False


class PipelineError(Exception):
    pass


def _sample_id_from_relative_path(rel_path: Path) -> str:
    stem_parts = list(rel_path.with_suffix("").parts)
    safe_parts = []
    for part in stem_parts:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in part.strip())
        cleaned = "_".join(tok for tok in cleaned.split("_") if tok)
        safe_parts.append(cleaned or "x")
    return "__".join(safe_parts)


def discover_mesh_files(input_root: Path, mesh_extensions: tuple[str, ...]) -> list[Path]:
    exts = {ext.lower() for ext in mesh_extensions}
    if not input_root.exists():
        raise PipelineError(f"Input root not found: {input_root}")

    files = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def _extract_labels(input_root: Path, mesh_path: Path) -> dict[str, str]:
    rel = mesh_path.relative_to(input_root)
    parts = rel.parts

    category = parts[0] if len(parts) >= 1 else "unknown"
    subcategory = parts[1] if len(parts) >= 2 else "unknown"
    object_folder = parts[2] if len(parts) >= 3 else mesh_path.stem

    return {
        "category": category,
        "subcategory": subcategory,
        "object_folder": object_folder,
    }


def _clean_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, validate=False)

    nondegenerate = mesh.nondegenerate_faces()
    if nondegenerate is not None:
        mesh.update_faces(nondegenerate)

    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    mesh.merge_vertices()

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise PipelineError("Mesh is empty after cleaning")

    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _split_count_triplet(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()

    raw = ratios * n
    base = np.floor(raw).astype(int)
    remainder = int(n - base.sum())

    frac_order = np.argsort(-(raw - base))
    for idx in frac_order[:remainder]:
        base[idx] += 1

    train_count, val_count, test_count = int(base[0]), int(base[1]), int(base[2])

    if val_count == 0:
        if train_count > test_count and train_count > 1:
            train_count -= 1
        else:
            test_count = max(test_count - 1, 0)
        val_count += 1

    if test_count == 0:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        else:
            val_count = max(val_count - 1, 0)
        test_count += 1

    if train_count <= 0:
        train_count = 1
        if val_count > test_count and val_count > 1:
            val_count -= 1
        else:
            test_count = max(test_count - 1, 0)

    if (train_count + val_count + test_count) != n:
        test_count = n - train_count - val_count

    return train_count, val_count, test_count


def stratified_split_records(
    records: list[dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    rng = np.random.default_rng(seed)

    by_category: dict[str, list[dict[str, object]]] = {}
    for record in records:
        category = str(record["category"])
        by_category.setdefault(category, []).append(record)

    train_records: list[dict[str, object]] = []
    val_records: list[dict[str, object]] = []
    test_records: list[dict[str, object]] = []

    for _, group in sorted(by_category.items()):
        idx = np.arange(len(group))
        rng.shuffle(idx)

        shuffled = [group[int(i)] for i in idx]
        train_count, val_count, test_count = _split_count_triplet(
            n=len(shuffled),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        train_records.extend(shuffled[:train_count])
        val_records.extend(shuffled[train_count: train_count + val_count])
        test_records.extend(shuffled[train_count + val_count: train_count + val_count + test_count])

    return {
        "train": sorted(train_records, key=lambda x: str(x["sample_id"])),
        "val": sorted(val_records, key=lambda x: str(x["sample_id"])),
        "test": sorted(test_records, key=lambda x: str(x["sample_id"])),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _zip_directory(source_dir: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(source_dir)
                zf.write(path, arcname=str(arcname.as_posix()))


def run_object_scanning_pipeline(config: ObjectScanningPipelineConfig) -> dict[str, object]:
    files = discover_mesh_files(config.input_root, config.mesh_extensions)
    if not files:
        raise PipelineError("No mesh files discovered for the configured extensions")

    preprocess_cfg = PreprocessConfig(target_abs_range=1.0, sort_vertices_zyx=True, sort_face_indices=False)
    reward_cfg = RewardConfig(run_self_intersection_check=config.include_self_intersection_check)

    output_mesh_dir = config.output_root / "meshes"
    manifests_dir = config.output_root / "manifests"
    reports_dir = config.output_root / "reports"

    output_mesh_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    valid_records: list[dict[str, object]] = []
    rejected_records: list[dict[str, object]] = []

    for mesh_path in files:
        rel = mesh_path.relative_to(config.input_root)
        sample_id = _sample_id_from_relative_path(rel)

        try:
            vertices, faces = load_mesh_arrays(mesh_path)
            vertices, faces = _clean_mesh(vertices, faces)
            processed = preprocess_mesh(vertices, faces, preprocess_cfg)

            processed_vertices = np.asarray(processed["vertices"], dtype=np.float32)
            processed_faces = np.asarray(processed["faces"], dtype=np.int32)

            if processed_vertices.shape[0] < config.min_vertices:
                raise PipelineError(f"Too few vertices: {processed_vertices.shape[0]}")
            if processed_faces.shape[0] < config.min_faces:
                raise PipelineError(f"Too few faces: {processed_faces.shape[0]}")

            metrics = evaluate_mesh_reward(processed_vertices, processed_faces, config=reward_cfg)

            npz_path = output_mesh_dir / f"{sample_id}.npz"
            np.savez_compressed(
                npz_path,
                vertices=processed_vertices,
                faces=processed_faces,
                source_relpath=str(rel.as_posix()),
            )

            labels = _extract_labels(config.input_root, mesh_path)
            valid_records.append(
                {
                    "sample_id": sample_id,
                    "source_relpath": str(rel.as_posix()),
                    "npz_relpath": str(npz_path.relative_to(config.output_root).as_posix()),
                    "category": labels["category"],
                    "subcategory": labels["subcategory"],
                    "object_folder": labels["object_folder"],
                    "num_vertices": int(processed_vertices.shape[0]),
                    "num_faces": int(processed_faces.shape[0]),
                    "watertight": bool(metrics["watertight"]),
                    "boundary_edge_ratio": float(metrics["boundary_edge_ratio"]),
                    "triangle_quality_score": float(metrics["triangle_quality_score"]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rejected_records.append(
                {
                    "source_relpath": str(rel.as_posix()),
                    "reason": str(exc),
                }
            )

    split_map = stratified_split_records(
        records=valid_records,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    all_manifest = {
        "num_samples": len(valid_records),
        "records": sorted(valid_records, key=lambda x: str(x["sample_id"])),
    }
    train_manifest = {
        "num_samples": len(split_map["train"]),
        "records": split_map["train"],
    }
    val_manifest = {
        "num_samples": len(split_map["val"]),
        "records": split_map["val"],
    }
    test_manifest = {
        "num_samples": len(split_map["test"]),
        "records": split_map["test"],
    }

    _write_json(manifests_dir / "all.json", all_manifest)
    _write_json(manifests_dir / "train.json", train_manifest)
    _write_json(manifests_dir / "val.json", val_manifest)
    _write_json(manifests_dir / "test.json", test_manifest)

    category_counts: dict[str, int] = {}
    for record in valid_records:
        category = str(record["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

    split_counts_by_category: dict[str, dict[str, int]] = {}
    for split_name, split_records in split_map.items():
        for record in split_records:
            category = str(record["category"])
            split_counts_by_category.setdefault(category, {"train": 0, "val": 0, "test": 0})
            split_counts_by_category[category][split_name] += 1

    summary = {
        "input_root": str(config.input_root),
        "output_root": str(config.output_root),
        "num_discovered": len(files),
        "num_processed": len(valid_records),
        "num_rejected": len(rejected_records),
        "split_counts": {
            "train": len(split_map["train"]),
            "val": len(split_map["val"]),
            "test": len(split_map["test"]),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "split_counts_by_category": dict(sorted(split_counts_by_category.items())),
        "rejection_rate": (len(rejected_records) / len(files)) if files else 0.0,
    }

    _write_json(reports_dir / "summary.json", summary)
    _write_json(reports_dir / "rejected.json", {"num_rejected": len(rejected_records), "records": rejected_records})

    readme_lines = [
        "Object Scanning Preprocessed Dataset",
        "",
        f"Input root: {config.input_root}",
        f"Processed samples: {len(valid_records)} / {len(files)}",
        f"Splits: train={len(split_map['train'])}, val={len(split_map['val'])}, test={len(split_map['test'])}",
        "",
        "Structure:",
        "- meshes/*.npz: normalized and cleaned mesh arrays (vertices, faces)",
        "- manifests/all.json: metadata for all accepted samples",
        "- manifests/train.json, val.json, test.json: split manifests",
        "- reports/summary.json: processing summary and counts",
        "- reports/rejected.json: per-file rejection reasons",
    ]
    (config.output_root / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    zip_path = config.output_root.parent / f"{config.output_root.name}.zip"
    _zip_directory(config.output_root, zip_path)

    result = {
        "summary": summary,
        "zip_path": str(zip_path),
        "output_root": str(config.output_root),
    }

    return result
