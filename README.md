# R4 Mesh RLVR (Local + VPS Workflow)

R4 is a reinforcement-learning-with-verifiable-rewards (RLVR) scaffold for improving mesh generation quality with deterministic geometry checks and constrained optimization.

This repository contains:
- Geometry reward and constraint modules (watertightness, boundary/quality metrics, optional self-intersection checks)
- Data preprocessing pipelines for local Object Scanning and ShapeNet GLB subsets
- Deterministic ShapeNet subset exploration and selective download tooling
- Offline GRPO-style training scaffold with step-level checkpointing and resume support
- Tests for core math/pipeline/checkpoint behavior

## Current Scope

This codebase currently provides the full data/validation/training scaffold and checkpoint-resume mechanics. A full GPU fine-tuning pipeline with MeshGPT policy loading is the next integration stage.

## Repository Layout

- `src/r4/`
  - `reward_engine.py`: deterministic reward + cost evaluation
  - `grpo.py`: group-relative advantage computation
  - `lagrangian.py`: PI-based Lagrangian multiplier controller
  - `trainer.py`: offline training scaffold with serializable state
  - `checkpointing.py`: checkpoint save/load/latest utilities
  - `data.py`, `object_scanning_pipeline.py`: preprocessing and split utilities
- `tools/`
  - `explore_shapenet_remote.py`: metadata-only exploration of gated ShapeNet repo
  - `prepare_shapenet_subset_manifest.py`: deterministic subset manifest creation
  - `download_shapenet_subset.py`: selective file download from manifest
  - `preprocess_shapenet_subset.py`: convert/split/zip subset
  - `preprocess_object_scanning.py`: preprocess local Object Scanning dataset
  - `generate_dataset_card.py`: produce dataset card JSON
  - `train_offline_grpo_with_checkpoints.py`: train loop with per-step checkpointing and resume
  - `upload_gso_to_vps.ps1`: upload and remote unzip helper
- `tests/`: pytest suite for core modules
- `Plan.md`, `PS.md`, `progress.md`: planning, spec, and milestone logs

## Git and Large File Policy

This repo intentionally excludes secrets and large assets via `.gitignore`:
- `.env` and other env variants
- generated datasets and artifacts (`data/`, `artifacts/`, `Object Scanning/`)
- large binary outputs (`*.zip`, `*.npz`, `*.ckpt`, etc.)

If you need to share data manifests/cards, place curated small metadata exports in a tracked docs/report folder outside ignored paths.

## Local Setup (Windows PowerShell)

Use the existing `.venv` in this project.

1. Activate environment:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```
2. Install/update dependencies:
```powershell
uv pip install -r requirements.txt
```
3. Run tests:
```powershell
uv run --python .\.venv\Scripts\python.exe -m pytest -q
```

Notes:
- `open3d` is optional and only installed for Python `< 3.13` as specified in `requirements.txt`.
- Self-intersection checking is available when Open3D is present.

## Environment Variables

Create `.env` from `.env.example`:

```bash
HUGGINGFACE_TOKEN=hf_your_token_here
```

Supported token keys in scripts:
- `HUGGINGFACE_TOKEN`
- `HUGGING_FACE_TOKEN`
- `HF_TOKEN`

## ShapeNet Subset Workflow (Deterministic)

### 1) Explore metadata only (no bulk download)
```powershell
uv run --python .\.venv\Scripts\python.exe tools\explore_shapenet_remote.py --env-file .env
```
Outputs:
- `artifacts/shapenet_exploration.json`
- `artifacts/shapenet_subset_options.json`

### 2) Generate deterministic subset manifest
```powershell
uv run --python .\.venv\Scripts\python.exe tools\prepare_shapenet_subset_manifest.py ^
  --env-file .env ^
  --bundle furniture_plus_display ^
  --cap-per-class 1000 ^
  --seed 42 ^
  --manifest-out artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json ^
  --paths-out artifacts/shapenet_subsets/furniture_plus_display_cap1000_paths.txt
```

### 3) Selective download from manifest
```powershell
uv run --python .\.venv\Scripts\python.exe tools\download_shapenet_subset.py ^
  --manifest artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json ^
  --env-file .env ^
  --output-dir data/shapenet_subsets/furniture_plus_display_cap1000 ^
  --workers 8
```

### 4) Preprocess and split
```powershell
.\.venv\Scripts\python.exe tools\preprocess_shapenet_subset.py ^
  --input-root data/shapenet_subsets/furniture_plus_display_cap1000 ^
  --output-root artifacts/shapenet_furniture_plus_display_cap1000_preprocessed ^
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 ^
  --seed 42 --min-vertices 16 --min-faces 16
```

Optional with Open3D check:
```powershell
.\.venv\Scripts\python.exe tools\preprocess_shapenet_subset.py --enable-self-intersection-check
```

### 5) Generate dataset card
```powershell
uv run --python .\.venv\Scripts\python.exe tools\generate_dataset_card.py ^
  --dataset-name shapenet_furniture_plus_display_cap1000_preprocessed ^
  --dataset-root artifacts/shapenet_furniture_plus_display_cap1000_preprocessed ^
  --summary artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/reports/summary.json ^
  --train artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/train.json ^
  --val artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/val.json ^
  --test artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/test.json ^
  --zip artifacts/shapenet_furniture_plus_display_cap1000_preprocessed.zip ^
  --output artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/dataset_card.json
```

## Offline Training with Per-Step Checkpointing

Run:
```powershell
uv run --python .\.venv\Scripts\python.exe tools\train_offline_grpo_with_checkpoints.py ^
  --train-manifest artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/train.json ^
  --val-manifest artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/val.json ^
  --test-manifest artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/test.json ^
  --dataset-root artifacts/shapenet_furniture_plus_display_cap1000_preprocessed ^
  --steps 500 --group-size 8 ^
  --checkpoint-dir artifacts/checkpoints/offline_grpo ^
  --save-every-steps 1 --keep-last 200 ^
  --eval-every-steps 100 --max-eval-groups 16 ^
  --run-name shapenet_exp01 --experiment-dir artifacts/experiments/offline_grpo
```

Default experiment artifacts now saved per run:
- `artifacts/experiments/offline_grpo/<run_name>/run_info.json`
- `artifacts/experiments/offline_grpo/<run_name>/train_timeline.jsonl`
- `artifacts/experiments/offline_grpo/<run_name>/eval_timeline.jsonl`
- `artifacts/experiments/offline_grpo/<run_name>/latest_metrics.json`

These files are chart-ready for experiment reporting.

Resume on same machine or a different machine (use same run name):
```powershell
uv run --python .\.venv\Scripts\python.exe tools\train_offline_grpo_with_checkpoints.py ^
  --steps 500 --group-size 8 ^
  --checkpoint-dir artifacts/checkpoints/offline_grpo ^
  --save-every-steps 1 --keep-last 200 ^
  --eval-every-steps 100 --max-eval-groups 16 ^
  --run-name shapenet_exp01 --experiment-dir artifacts/experiments/offline_grpo ^
  --resume
```

Optional: disable periodic val/test evaluation logs with `--disable-eval`.

## VPS Setup and Operations Guide

A complete VPS runbook is available at:
- `VPS.md` (single-file whole-picture workflow)
- `docs/VPS_SETUP.md`

It includes:
- server provisioning and hardening
- GPU/runtime verification
- data transfer strategies
- long-running job orchestration with `tmux`
- checkpoint sync and cross-machine resume procedures
- recovery playbooks for preemption/failures

## Quick Health Checks

1. Import sanity:
```powershell
uv run --python .\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); import r4; print('ok')"
```
2. Reward smoke test:
```powershell
uv run --python .\.venv\Scripts\python.exe tools\smoke_reward_checks.py
```
3. Checkpointing tests:
```powershell
uv run --python .\.venv\Scripts\python.exe -m pytest -q tests\test_checkpointing.py
```

## License and Data Terms

- Respect the ShapeNet license and gated access terms.
- Do not commit proprietary datasets, model weights, tokens, or generated large artifacts.
