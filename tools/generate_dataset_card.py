from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _category_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        category = str(r.get("category", "unknown"))
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _to_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_card(
    dataset_name: str,
    dataset_root: Path,
    summary_path: Path,
    train_manifest_path: Path,
    val_manifest_path: Path,
    test_manifest_path: Path,
    zip_path: Path,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    train_manifest = _read_json(train_manifest_path)
    val_manifest = _read_json(val_manifest_path)
    test_manifest = _read_json(test_manifest_path)

    train_records = train_manifest.get("records", [])
    val_records = val_manifest.get("records", [])
    test_records = test_manifest.get("records", [])

    total = int(summary.get("num_processed", 0))
    split_counts = {
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
    }

    split_ratios = {
        k: (v / total if total > 0 else 0.0)
        for k, v in split_counts.items()
    }

    zip_size = zip_path.stat().st_size
    zip_hash = _sha256(zip_path)

    card = {
        "dataset_name": dataset_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "input_root": summary.get("input_root"),
            "num_discovered": int(summary.get("num_discovered", 0)),
            "num_processed": int(summary.get("num_processed", 0)),
            "num_rejected": int(summary.get("num_rejected", 0)),
            "rejection_rate": float(summary.get("rejection_rate", 0.0)),
        },
        "splits": {
            "counts": split_counts,
            "ratios": split_ratios,
            "category_counts": {
                "train": _category_counts(train_records),
                "val": _category_counts(val_records),
                "test": _category_counts(test_records),
                "all": dict(summary.get("category_counts", {})),
            },
        },
        "artifacts": {
            "dataset_root": _to_rel(dataset_root, Path.cwd()),
            "summary_json": _to_rel(summary_path, Path.cwd()),
            "train_manifest_json": _to_rel(train_manifest_path, Path.cwd()),
            "val_manifest_json": _to_rel(val_manifest_path, Path.cwd()),
            "test_manifest_json": _to_rel(test_manifest_path, Path.cwd()),
            "zip_path": _to_rel(zip_path, Path.cwd()),
            "zip_size_bytes": zip_size,
            "zip_sha256": zip_hash,
        },
    }

    return card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact dataset card JSON for preprocessed GSO dataset")
    parser.add_argument("--dataset-name", default="gso_preprocessed", help="Dataset name in card")
    parser.add_argument("--dataset-root", default="artifacts/gso_preprocessed", help="Processed dataset root")
    parser.add_argument("--summary", default="artifacts/gso_preprocessed/reports/summary.json", help="Summary JSON path")
    parser.add_argument("--train", default="artifacts/gso_preprocessed/manifests/train.json", help="Train manifest path")
    parser.add_argument("--val", default="artifacts/gso_preprocessed/manifests/val.json", help="Val manifest path")
    parser.add_argument("--test", default="artifacts/gso_preprocessed/manifests/test.json", help="Test manifest path")
    parser.add_argument("--zip", default="artifacts/gso_preprocessed.zip", help="Zipped dataset path")
    parser.add_argument("--output", default="artifacts/gso_preprocessed/dataset_card.json", help="Output card path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    summary_path = Path(args.summary)
    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    zip_path = Path(args.zip)
    output_path = Path(args.output)

    card = build_card(
        dataset_name=args.dataset_name,
        dataset_root=dataset_root,
        summary_path=summary_path,
        train_manifest_path=train_path,
        val_manifest_path=val_path,
        test_manifest_path=test_path,
        zip_path=zip_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    print(f"Wrote dataset card: {output_path}")


if __name__ == "__main__":
    main()
