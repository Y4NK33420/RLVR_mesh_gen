from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import socket
import subprocess
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


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


def _controller_snapshot(trainer: OfflineGRPOTrainer) -> dict[str, float | None]:
    return {
        "lambda_value": float(trainer.controller.lambda_value),
        "kp": float(trainer.controller.kp),
        "ki": float(trainer.controller.ki),
        "target_cost": float(trainer.controller.target_cost),
        "min_lambda": float(trainer.controller.min_lambda),
        "max_lambda": (None if trainer.controller.max_lambda is None else float(trainer.controller.max_lambda)),
        "integral_error": float(trainer.controller._integral_error),
    }


def _controller_restore(trainer: OfflineGRPOTrainer, snapshot: dict[str, float | None]) -> None:
    trainer.controller.lambda_value = float(snapshot["lambda_value"])
    trainer.controller.kp = float(snapshot["kp"])
    trainer.controller.ki = float(snapshot["ki"])
    trainer.controller.target_cost = float(snapshot["target_cost"])
    trainer.controller.min_lambda = float(snapshot["min_lambda"])
    trainer.controller.max_lambda = None if snapshot["max_lambda"] is None else float(snapshot["max_lambda"])
    trainer.controller._integral_error = float(snapshot["integral_error"])


def _evaluate_split(
    trainer: OfflineGRPOTrainer,
    groups: list[list[dict[str, Any]]],
    dataset_root: Path,
    split_name: str,
    step: int,
    max_eval_groups: int,
) -> dict[str, Any] | None:
    if not groups:
        return None

    limit = len(groups) if max_eval_groups <= 0 else min(len(groups), max_eval_groups)

    eval_rows: list[dict[str, float]] = []
    for idx in range(limit):
        group = groups[idx]
        meshes = _load_mesh_group(group, dataset_root=dataset_root)

        # Keep validation/testing read-only with respect to controller state.
        snapshot = _controller_snapshot(trainer)
        result = trainer.evaluate_group(meshes)
        _controller_restore(trainer, snapshot)

        eval_rows.append(
            {
                "mean_reward": float(result["mean_reward"]),
                "mean_cost": float(result["mean_cost"]),
                "lagrangian_objective": float(result["lagrangian_objective"]),
            }
        )

    if not eval_rows:
        return None

    return {
        "timestamp_utc": _utc_now_iso(),
        "step": int(step),
        "split": split_name,
        "groups_available": int(len(groups)),
        "groups_evaluated": int(limit),
        "mean_reward": float(np.mean([x["mean_reward"] for x in eval_rows])),
        "mean_cost": float(np.mean([x["mean_cost"] for x in eval_rows])),
        "mean_lagrangian_objective": float(np.mean([x["lagrangian_objective"] for x in eval_rows])),
        "lambda_reference": float(trainer.controller.lambda_value),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline GRPO loop with step-level checkpointing and resume")
    parser.add_argument(
        "--train-manifest",
        default="artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/train.json",
        help="Path to train manifest JSON",
    )
    parser.add_argument(
        "--val-manifest",
        default="artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/val.json",
        help="Path to validation manifest JSON",
    )
    parser.add_argument(
        "--test-manifest",
        default="artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/test.json",
        help="Path to test manifest JSON",
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
    parser.add_argument("--keep-last", type=int, default=200, help="How many recent checkpoints to keep")
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=100,
        help="Run validation/test evaluation every N training steps",
    )
    parser.add_argument(
        "--max-eval-groups",
        type=int,
        default=16,
        help="Max groups per split during each evaluation event (0 means all groups)",
    )
    parser.add_argument(
        "--disable-eval",
        action="store_true",
        help="Disable periodic validation/test evaluation logging",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional experiment run name. If omitted, one is generated or recovered on resume.",
    )
    parser.add_argument(
        "--experiment-dir",
        default="artifacts/experiments/offline_grpo",
        help="Directory to store per-run metadata and timeline logs",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument(
        "--metrics-out",
        default="",
        help="Path to write final metrics summary. If empty, writes to <run_dir>/latest_metrics.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)
    test_manifest = Path(args.test_manifest)
    dataset_root = Path(args.dataset_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    experiment_dir = Path(args.experiment_dir)

    train_records = _load_manifest(train_manifest)
    train_groups = _group_records(train_records, group_size=args.group_size)

    trainer = OfflineGRPOTrainer(train_config=TrainConfig(group_size=args.group_size))

    loaded_checkpoint: dict[str, Any] | None = None
    start_step = 0
    if args.resume:
        loaded_checkpoint = load_latest_training_checkpoint(checkpoint_dir=checkpoint_dir, trainer=trainer)
        if loaded_checkpoint is not None:
            start_step = int(loaded_checkpoint.get("step", trainer.global_step))
            print(f"Resumed from checkpoint step {start_step}")

    if args.steps <= 0:
        raise ValueError("steps must be > 0")

    recovered_run_name = ""
    if loaded_checkpoint is not None:
        recovered_run_name = str((loaded_checkpoint.get("extra_state") or {}).get("run_name", ""))

    run_name = args.run_name.strip() or recovered_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = experiment_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_timeline_path = run_dir / "train_timeline.jsonl"
    eval_timeline_path = run_dir / "eval_timeline.jsonl"
    run_info_path = run_dir / "run_info.json"

    metrics_out = Path(args.metrics_out) if args.metrics_out.strip() else (run_dir / "latest_metrics.json")

    eval_groups: dict[str, list[list[dict[str, Any]]]] = {}
    if not args.disable_eval:
        for split_name, split_manifest in (("val", val_manifest), ("test", test_manifest)):
            if not split_manifest.exists():
                print(f"Skipping {split_name} eval: manifest missing at {split_manifest.as_posix()}")
                continue
            try:
                split_records = _load_manifest(split_manifest)
                split_groups = _group_records(split_records, group_size=args.group_size)
                eval_groups[split_name] = split_groups
            except ValueError as exc:
                print(f"Skipping {split_name} eval: {exc}")

    run_info = {
        "run_name": run_name,
        "created_at_utc": _utc_now_iso(),
        "resumed": bool(args.resume),
        "resume_start_step": int(start_step),
        "git_commit": _git_commit(ROOT),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "train_manifest": train_manifest.as_posix(),
        "val_manifest": val_manifest.as_posix(),
        "test_manifest": test_manifest.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "experiment_dir": experiment_dir.as_posix(),
        "run_dir": run_dir.as_posix(),
        "train_timeline_jsonl": train_timeline_path.as_posix(),
        "eval_timeline_jsonl": eval_timeline_path.as_posix(),
        "metrics_out": metrics_out.as_posix(),
        "args": vars(args),
        "train_group_count": int(len(train_groups)),
        "eval_group_counts": {k: int(len(v)) for k, v in eval_groups.items()},
    }
    run_info_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    metrics: list[dict[str, Any]] = []
    latest_eval: dict[str, dict[str, Any]] = {}

    for local_step in range(args.steps):
        global_idx = start_step + local_step
        group = train_groups[global_idx % len(train_groups)]
        meshes = _load_mesh_group(group, dataset_root=dataset_root)

        result = trainer.step(meshes)
        step_payload = {
            "step": int(result["step"]),
            "mean_reward": float(result["mean_reward"]),
            "mean_cost": float(result["mean_cost"]),
            "lambda_after": float(result["lambda_after"]),
            "lagrangian_objective": float(result["lagrangian_objective"]),
        }
        metrics.append(step_payload)

        _append_jsonl(
            train_timeline_path,
            {
                "timestamp_utc": _utc_now_iso(),
                "run_name": run_name,
                "step": int(result["step"]),
                "mean_reward": float(result["mean_reward"]),
                "mean_cost": float(result["mean_cost"]),
                "lambda_before": float(result["lambda_before"]),
                "lambda_after": float(result["lambda_after"]),
                "lagrangian_objective": float(result["lagrangian_objective"]),
                "group_size": int(result["group_size"]),
            },
        )

        if (result["step"] % args.save_every_steps) == 0:
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=int(result["step"]),
                trainer=trainer,
                extra_state={
                    "run_name": run_name,
                    "run_dir": run_dir.as_posix(),
                    "train_timeline_jsonl": train_timeline_path.as_posix(),
                    "eval_timeline_jsonl": eval_timeline_path.as_posix(),
                    "train_manifest": str(train_manifest.as_posix()),
                    "dataset_root": str(dataset_root.as_posix()),
                    "group_size": int(args.group_size),
                    "last_metrics": step_payload,
                },
                keep_last_n=int(args.keep_last),
            )

        if (not args.disable_eval) and args.eval_every_steps > 0 and (int(result["step"]) % args.eval_every_steps) == 0:
            for split_name, split_groups in eval_groups.items():
                eval_payload = _evaluate_split(
                    trainer=trainer,
                    groups=split_groups,
                    dataset_root=dataset_root,
                    split_name=split_name,
                    step=int(result["step"]),
                    max_eval_groups=int(args.max_eval_groups),
                )
                if eval_payload is not None:
                    latest_eval[split_name] = eval_payload
                    _append_jsonl(eval_timeline_path, eval_payload)

    if metrics:
        final_payload = {
            "run_name": run_name,
            "run_dir": run_dir.as_posix(),
            "train_timeline_jsonl": train_timeline_path.as_posix(),
            "eval_timeline_jsonl": eval_timeline_path.as_posix(),
            "start_step": int(start_step),
            "end_step": int(metrics[-1]["step"]),
            "steps_run": int(len(metrics)),
            "last": metrics[-1],
            "mean_reward_over_run": float(np.mean([m["mean_reward"] for m in metrics])),
            "mean_cost_over_run": float(np.mean([m["mean_cost"] for m in metrics])),
        }
        if latest_eval:
            final_payload["latest_eval"] = latest_eval
    else:
        final_payload = {
            "run_name": run_name,
            "run_dir": run_dir.as_posix(),
            "train_timeline_jsonl": train_timeline_path.as_posix(),
            "eval_timeline_jsonl": eval_timeline_path.as_posix(),
            "start_step": int(start_step),
            "end_step": int(start_step),
            "steps_run": 0,
        }

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    run_info_update = json.loads(run_info_path.read_text(encoding="utf-8"))
    run_info_update["ended_at_utc"] = _utc_now_iso()
    run_info_update["final_step"] = int(final_payload.get("end_step", start_step))
    run_info_path.write_text(json.dumps(run_info_update, indent=2), encoding="utf-8")

    print(json.dumps(final_payload, indent=2))


if __name__ == "__main__":
    main()
