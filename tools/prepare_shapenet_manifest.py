from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r4.data import find_shapenet_meshes, sample_paths, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a sampled ShapeNet manifest for chair meshes.")
    parser.add_argument("--dataset-root", required=True, help="Path to ShapeNet root directory")
    parser.add_argument("--output", default="artifacts/chair_manifest.json", help="Output manifest path")
    parser.add_argument("--synset", default="03001627", help="ShapeNet synset id (chairs by default)")
    parser.add_argument("--limit", type=int, default=1000, help="Number of mesh paths to sample")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_paths = find_shapenet_meshes(args.dataset_root, synset_id=args.synset)
    sampled = sample_paths(all_paths, limit=args.limit, seed=args.seed)
    write_manifest(sampled, args.output)

    print(f"Found {len(all_paths)} meshes under synset {args.synset}")
    print(f"Wrote sampled manifest with {len(sampled)} entries to {args.output}")


if __name__ == "__main__":
    main()
