from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO = "ShapeNet/shapenetcore-glb"


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


def build_exploration(repo_id: str, token: str) -> dict[str, object]:
    api = HfApi(token=token)
    info = api.dataset_info(repo_id=repo_id, files_metadata=True)

    class_stats: dict[str, dict[str, int]] = {}
    total_glb = 0
    total_bytes = 0

    for sibling in info.siblings or []:
        path = getattr(sibling, "rfilename", None) or ""
        if not path.lower().endswith(".glb") or "/" not in path:
            continue

        cls = path.split("/")[0]
        size = int(getattr(sibling, "size", 0) or 0)

        entry = class_stats.setdefault(cls, {"count": 0, "bytes": 0})
        entry["count"] += 1
        entry["bytes"] += size

        total_glb += 1
        total_bytes += size

    classes = [
        {"class": key, "count": value["count"], "bytes": value["bytes"]}
        for key, value in class_stats.items()
    ]

    return {
        "repo": repo_id,
        "sha": getattr(info, "sha", None),
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "last_modified": str(getattr(info, "lastModified", None)),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "total_glb": total_glb,
        "total_bytes": total_bytes,
        "class_count": len(classes),
        "classes_by_count": sorted(classes, key=lambda x: x["count"], reverse=True),
        "classes_by_size": sorted(classes, key=lambda x: x["bytes"], reverse=True),
    }


def _estimate_bundle(
    class_map: dict[str, dict[str, int]],
    classes: list[str],
    cap: int | None,
) -> dict[str, object]:
    total_count = 0
    total_bytes = 0.0

    for cls in classes:
        if cls not in class_map:
            continue
        c = class_map[cls]["count"]
        b = class_map[cls]["bytes"]
        take = c if cap is None else min(cap, c)
        avg = (b / c) if c > 0 else 0.0

        total_count += take
        total_bytes += avg * take

    return {
        "classes": classes,
        "cap_per_class": cap,
        "estimated_samples": total_count,
        "estimated_bytes": int(total_bytes),
        "estimated_gb": round(total_bytes / 1e9, 3),
    }


def build_subset_options(exploration: dict[str, object]) -> dict[str, object]:
    classes = exploration["classes_by_count"]
    class_map = {entry["class"]: entry for entry in classes}

    bundles = {
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

    options: dict[str, object] = {}
    for name, bundle_classes in bundles.items():
        options[name] = {
            "full": _estimate_bundle(class_map, bundle_classes, cap=None),
            "cap_500": _estimate_bundle(class_map, bundle_classes, cap=500),
            "cap_1000": _estimate_bundle(class_map, bundle_classes, cap=1000),
            "cap_1500": _estimate_bundle(class_map, bundle_classes, cap=1500),
        }

    return {
        "repo": exploration["repo"],
        "generated_from_sha": exploration["sha"],
        "options": options,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore ShapeNet GLB repository metadata without downloading meshes")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Dataset repo id")
    parser.add_argument("--env-file", default=".env", help="Path to env file with Hugging Face token")
    parser.add_argument("--exploration-out", default="artifacts/shapenet_exploration.json", help="Output exploration JSON")
    parser.add_argument("--options-out", default="artifacts/shapenet_subset_options.json", help="Output subset options JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = _resolve_token(Path(args.env_file))
    exploration = build_exploration(repo_id=args.repo, token=token)
    options = build_subset_options(exploration)

    exploration_path = Path(args.exploration_out)
    options_path = Path(args.options_out)

    exploration_path.parent.mkdir(parents=True, exist_ok=True)
    options_path.parent.mkdir(parents=True, exist_ok=True)

    exploration_path.write_text(json.dumps(exploration, indent=2), encoding="utf-8")
    options_path.write_text(json.dumps(options, indent=2), encoding="utf-8")

    print(f"wrote exploration: {exploration_path.as_posix()}")
    print(f"wrote options: {options_path.as_posix()}")
    print(f"total_glb: {exploration['total_glb']}")
    print(f"total_size_gb: {round(exploration['total_bytes'] / 1e9, 3)}")


if __name__ == "__main__":
    main()
