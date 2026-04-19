from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import pickle
from pathlib import Path
from typing import Any

from .trainer import OfflineGRPOTrainer


def _save_blob(path: Path, payload: dict[str, Any]) -> str:
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        import torch

        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        return "torch"
    except Exception:
        with tmp_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
        return "pickle"


def _load_blob(path: Path) -> dict[str, Any]:
    try:
        import torch

        return torch.load(path, map_location="cpu")
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    return checkpoint_dir / f"step_{step:08d}.ckpt"


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    step: int,
    trainer: OfflineGRPOTrainer,
    model_state: dict[str, Any] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
    keep_last_n: int = 5,
) -> Path:
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = _checkpoint_path(out_dir, step)
    payload: dict[str, Any] = {
        "step": int(step),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "trainer_state": trainer.state_dict(),
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "extra_state": (extra_state or {}),
    }

    serialization = _save_blob(ckpt_path, payload)

    latest_payload = {
        "latest_checkpoint": ckpt_path.name,
        "step": int(step),
        "saved_at_utc": payload["saved_at_utc"],
        "serialization": serialization,
    }
    _atomic_write_json(out_dir / "latest.json", latest_payload)

    checkpoints = sorted(out_dir.glob("step_*.ckpt"))
    if keep_last_n > 0 and len(checkpoints) > keep_last_n:
        to_delete = checkpoints[: len(checkpoints) - keep_last_n]
        for old_path in to_delete:
            old_path.unlink(missing_ok=True)

    return ckpt_path


def load_training_checkpoint(
    checkpoint_path: str | Path,
    trainer: OfflineGRPOTrainer | None = None,
    model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    payload = _load_blob(path)

    if trainer is not None and payload.get("trainer_state") is not None:
        trainer.load_state_dict(payload["trainer_state"])

    if model is not None and payload.get("model_state") is not None:
        model.load_state_dict(payload["model_state"])

    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])

    return payload


def load_latest_training_checkpoint(
    checkpoint_dir: str | Path,
    trainer: OfflineGRPOTrainer | None = None,
    model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any] | None:
    ckpt_dir = Path(checkpoint_dir)
    latest_path = ckpt_dir / "latest.json"

    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        ckpt_file = latest.get("latest_checkpoint")
        if ckpt_file:
            return load_training_checkpoint(
                checkpoint_path=ckpt_dir / str(ckpt_file),
                trainer=trainer,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    checkpoints = sorted(ckpt_dir.glob("step_*.ckpt"))
    if not checkpoints:
        return None

    return load_training_checkpoint(
        checkpoint_path=checkpoints[-1],
        trainer=trainer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
