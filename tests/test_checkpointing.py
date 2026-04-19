from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from r4.checkpointing import load_latest_training_checkpoint, save_training_checkpoint
from r4.trainer import OfflineGRPOTrainer


def _mesh_pair() -> list[tuple[np.ndarray, np.ndarray]]:
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    open_mesh = box.copy()
    open_mesh.faces = open_mesh.faces[:-1]

    return [
        (np.asarray(box.vertices), np.asarray(box.faces)),
        (np.asarray(open_mesh.vertices), np.asarray(open_mesh.faces)),
    ]


def test_checkpoint_save_and_resume_roundtrip(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"

    trainer = OfflineGRPOTrainer()
    meshes = _mesh_pair()

    trainer.step(meshes)
    trainer.step(meshes)

    saved_path = save_training_checkpoint(
        checkpoint_dir=ckpt_dir,
        step=trainer.global_step,
        trainer=trainer,
        extra_state={"note": "roundtrip"},
        keep_last_n=3,
    )

    assert saved_path.exists()

    resumed = OfflineGRPOTrainer()
    payload = load_latest_training_checkpoint(checkpoint_dir=ckpt_dir, trainer=resumed)

    assert payload is not None
    assert resumed.global_step == trainer.global_step
    assert np.isclose(resumed.controller.lambda_value, trainer.controller.lambda_value)
    assert np.isclose(resumed.controller._integral_error, trainer.controller._integral_error)


def test_checkpoint_prunes_old_files(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpts"
    trainer = OfflineGRPOTrainer()
    meshes = _mesh_pair()

    for _ in range(5):
        trainer.step(meshes)
        save_training_checkpoint(
            checkpoint_dir=ckpt_dir,
            step=trainer.global_step,
            trainer=trainer,
            keep_last_n=2,
        )

    checkpoints = sorted(ckpt_dir.glob("step_*.ckpt"))
    assert len(checkpoints) == 2
