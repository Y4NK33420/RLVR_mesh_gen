from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r4.object_scanning_pipeline import (  # noqa: E402
    ObjectScanningPipelineConfig,
    run_object_scanning_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess and split Google Scanned Objects dataset")
    parser.add_argument("--input-root", default="Object Scanning", help="Input dataset root")
    parser.add_argument("--output-root", default="artifacts/gso_preprocessed", help="Output dataset root")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for partitioning")
    parser.add_argument("--min-vertices", type=int, default=16, help="Minimum vertices per accepted mesh")
    parser.add_argument("--min-faces", type=int, default=16, help="Minimum faces per accepted mesh")
    parser.add_argument(
        "--enable-self-intersection-check",
        action="store_true",
        help="Enable Open3D self-intersection checks if Open3D is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ObjectScanningPipelineConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_vertices=args.min_vertices,
        min_faces=args.min_faces,
        include_self_intersection_check=args.enable_self_intersection_check,
    )

    result = run_object_scanning_pipeline(cfg)

    print("Preprocessing complete")
    print(json.dumps(result["summary"], indent=2))
    print(f"Output root: {result['output_root']}")
    print(f"Zip file: {result['zip_path']}")


if __name__ == "__main__":
    main()
