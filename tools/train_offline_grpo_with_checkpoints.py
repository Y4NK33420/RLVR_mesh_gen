from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from r4.checkpointing import load_latest_training_checkpoint, save_training_checkpoint
from r4.trainer import OfflineGRPOTrainer, TrainConfig


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No records found in manifest: {path}")
    return records


def _group_records(records: list[dict[str, Any]], group_size: int) -> list[list[dict[str, Any]]]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    groups: list[list[dict[str, Any]]] = []
    for i in range(0, len(records), group_size):
        chunk = records[i : i + group_size]
        if len(chunk) == group_size:
            groups.append(chunk)

    if not groups:
        raise ValueError("Not enough samples to form at least one full group")

    return groups


def _load_mesh_group(group: list[dict[str, Any]], dataset_root: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    meshes: list[tuple[np.ndarray, np.ndarray]] = []

    for record in group:
        npz_relpath = record.get("npz_relpath")
        if not isinstance(npz_relpath, str):
            raise ValueError("Manifest record is missing string npz_relpath")

        npz_path = dataset_root / npz_relpath
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        meshes.append((data["vertices"], data["faces"]))

    return meshes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline GRPO loop with step-level checkpointing and resume")
    parser.add_argument(
        "--train-manifest",
        default="artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/train.json",
        help="Path to train manifest JSON",
    )
    parser.add_argument(
        "--dataset-root",
        default="artifacts/shapenet_furniture_plus_display_cap1000_preprocessed",
        help="Root directory containing NPZ files",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of training steps to run in this invocation",
    )
    parser.add_argument("--group-size", type=int, default=8, help="Group size for GRPO-style rollout")
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/checkpoints/offline_grpo",
        help="Directory to store checkpoints",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=1,
        help="Checkpoint frequency in steps (1 means every step)",
    )
    parser.add_argument("--keep-last", type=int, default=5, help="How many recent checkpoints to keep")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument(
        "--metrics-out",
        default="artifacts/checkpoints/offline_grpo/latest_metrics.json",
        help="Path to write final metrics summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_manifest = Path(args.train_manifest)
    dataset_root = Path(args.dataset_root)
    checkpoint_dir = Path(args.checkpoint_dir)

    records = _load_manifest(train_manifest)
    groups = _group_records(records, group_size=args.group_size)

    trainer = OfflineGRPOTrainer(train_config=TrainConfig(group_size=args.group_size))

    start_step = 0
    if args.resume:
        loaded = load_latest_training_checkpoint(checkpoint_dir=checkpoint_dir, trainer=trainer)
        if loaded is not None:
            start_step = int(loaded.get("step", trainer.global_step))
            print(f"Resumed from checkpoint step {start_step}")

    if args.steps <= 0:
        raise ValueError("steps must be > 0")

    metrics: list[dict[str, Any]] = []

    for local_step in range(args.steps):
        global_idx = start_step + local_step
        group = groups[global_idx % len(groups)]
        meshes = _load_mesh_group(group, dataset_root=dataset_root)

        result = trainer.step(meshes)
        metrics.append(
            {
                "step": int(result["step"]),
                "mean_reward": float(result["mean_reward"]),
                "mean_cost": float(result["mean_cost"]),
                "lambda_after": float(result["lambda_after"]),
                "lagrangian_objective": float(result["lagrangian_objective"]),
            }
        )

        if (result["step"] % args.save_every_steps) == 0:
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=int(result["step"]),
                trainer=trainer,
                extra_state={
                    "train_manifest": str(train_manifest.as_posix()),
                    "dataset_root": str(dataset_root.as_posix()),
                    "group_size": int(args.group_size),
                    "last_metrics": metrics[-1],
                },
                keep_last_n=int(args.keep_last),
            )

    if metrics:
        final_payload = {
            "start_step": int(start_step),
            "end_step": int(metrics[-1]["step"]),
            "steps_run": int(len(metrics)),
            "last": metrics[-1],
            "mean_reward_over_run": float(np.mean([m["mean_reward"] for m in metrics])),
            "mean_cost_over_run": float(np.mean([m["mean_cost"] for m in metrics])),
        }
    else:
        final_payload = {
            "start_step": int(start_step),
            "end_step": int(start_step),
            "steps_run": 0,
        }

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    print(json.dumps(final_payload, indent=2))


if __name__ == "__main__":
    main()
