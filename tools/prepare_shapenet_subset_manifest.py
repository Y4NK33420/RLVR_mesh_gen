from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


BUNDLES: dict[str, list[str]] = {
    "furniture_core": ["chair", "table", "sofa", "cabinet", "bench", "bookshelf", "bed", "lamp"],
    "furniture_plus_display": [
        "chair",
        "table",
        "sofa",
        "cabinet",
        "bench",
        "bookshelf",
        "bed",
        "lamp",
        "display",
        "loudspeaker",
    ],
    "balanced10": ["chair", "table", "car", "airplane", "sofa", "lamp", "bench", "cabinet", "display", "loudspeaker"],
    "transport_core": ["car", "airplane", "bus", "motorcycle", "train"],
}


def _resolve_token(env_path: Path | None) -> str:
    env_candidates = ["HUGGINGFACE_TOKEN", "HUGGING_FACE_TOKEN", "HF_TOKEN"]

    for key in env_candidates:
        value = os.environ.get(key)
        if value:
            return value

    if env_path is not None and env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key in env_candidates and value.strip():
                return value.strip()

    raise RuntimeError(
        "No Hugging Face token found. Set HUGGINGFACE_TOKEN or HUGGING_FACE_TOKEN in environment or .env."
    )


def _sample_records(records: list[dict[str, Any]], cap: int, seed: int, class_index: int) -> list[dict[str, Any]]:
    if cap <= 0 or len(records) <= cap:
        return records

    rng = random.Random(seed + (class_index * 1009))
    idx = list(range(len(records)))
    selected_idx = set(rng.sample(idx, k=cap))

    selected = [records[i] for i in range(len(records)) if i in selected_idx]
    selected.sort(key=lambda x: str(x["path"]))
    return selected


def build_manifest(repo: str, bundle_name: str, cap_per_class: int, seed: int, token: str) -> dict[str, Any]:
    if bundle_name not in BUNDLES:
        raise ValueError(f"Unknown bundle '{bundle_name}'. Available: {sorted(BUNDLES)}")

    target_classes = BUNDLES[bundle_name]
    api = HfApi(token=token)
    info = api.dataset_info(repo_id=repo, files_metadata=True)

    all_by_class: dict[str, list[dict[str, Any]]] = {cls: [] for cls in target_classes}

    for sibling in info.siblings or []:
        path = getattr(sibling, "rfilename", None) or ""
        if not path.lower().endswith(".glb") or "/" not in path:
            continue

        cls = path.split("/")[0]
        if cls not in all_by_class:
            continue

        size = int(getattr(sibling, "size", 0) or 0)
        all_by_class[cls].append({"path": path, "class": cls, "bytes": size})

    selected_records: list[dict[str, Any]] = []
    class_stats: list[dict[str, Any]] = []

    for i, cls in enumerate(target_classes):
        available = sorted(all_by_class[cls], key=lambda x: str(x["path"]))
        chosen = _sample_records(available, cap=cap_per_class, seed=seed, class_index=i)

        selected_records.extend(chosen)
        class_stats.append(
            {
                "class": cls,
                "available": len(available),
                "selected": len(chosen),
                "available_bytes": int(sum(x["bytes"] for x in available)),
                "selected_bytes": int(sum(x["bytes"] for x in chosen)),
            }
        )

    selected_records.sort(key=lambda x: (str(x["class"]), str(x["path"])))

    manifest: dict[str, Any] = {
        "repo": repo,
        "repo_sha": getattr(info, "sha", None),
        "gated": getattr(info, "gated", None),
        "selection": {
            "bundle": bundle_name,
            "classes": target_classes,
            "cap_per_class": cap_per_class,
            "seed": seed,
            "strategy": "random_without_replacement_per_class",
        },
        "summary": {
            "selected_files": len(selected_records),
            "selected_bytes": int(sum(x["bytes"] for x in selected_records)),
            "selected_gb": round(sum(x["bytes"] for x in selected_records) / 1e9, 3),
        },
        "class_stats": class_stats,
        "records": selected_records,
    }

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic ShapeNet subset manifest from metadata")
    parser.add_argument("--repo", default="ShapeNet/shapenetcore-glb", help="Dataset repo")
    parser.add_argument("--env-file", default=".env", help="Path to .env token file")
    parser.add_argument("--bundle", default="furniture_plus_display", choices=sorted(BUNDLES.keys()))
    parser.add_argument("--cap-per-class", type=int, default=1000, help="Max files to sample per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--manifest-out",
        default="artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json",
        help="Output JSON manifest path",
    )
    parser.add_argument(
        "--paths-out",
        default="artifacts/shapenet_subsets/furniture_plus_display_cap1000_paths.txt",
        help="Output plain path list for the selected files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = _resolve_token(Path(args.env_file))
    manifest = build_manifest(
        repo=args.repo,
        bundle_name=args.bundle,
        cap_per_class=args.cap_per_class,
        seed=args.seed,
        token=token,
    )

    manifest_out = Path(args.manifest_out)
    paths_out = Path(args.paths_out)

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    paths_out.parent.mkdir(parents=True, exist_ok=True)

    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths_out.write_text("\n".join(x["path"] for x in manifest["records"]) + "\n", encoding="utf-8")

    print(f"wrote manifest: {manifest_out.as_posix()}")
    print(f"wrote path list: {paths_out.as_posix()}")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
