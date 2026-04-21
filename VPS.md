# VPS Execution Guide (Whole Picture)

This document is the single end-to-end guide for running this project on a VPS.

## 1) What We Are Trying To Do

Goal for this project version:
- Use RLVR-style mesh quality alignment with deterministic geometry rewards.
- Train the offline GRPO scaffold with step-level checkpointing so runs can survive machine switching or preemption.
- Persist chart-ready experiment timelines and run metadata for final project reporting.
- Keep topology constraints and reward metrics reproducible.

Practical objective right now:
- Use the ShapeNet GLB subset workflow for this version.
- Prepare a train-ready preprocessed dataset.
- Start training on VPS with resumable checkpoints.

## 2) Dataset Choice For This Version

This version uses ShapeNet GLB subset selection:
- Repository: ShapeNet/shapenetcore-glb
- Bundle used: furniture_plus_display
- Cap per class: 1000
- Typical outputs:
  - subset manifest and path list under artifacts/shapenet_subsets/
  - downloaded meshes under data/shapenet_subsets/
  - preprocessed dataset under artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/

## 3) Files You Will Use Most

Core training and checkpointing:
- tools/train_offline_grpo_with_checkpoints.py
- src/r4/trainer.py
- src/r4/checkpointing.py

ShapeNet workflow:
- tools/explore_shapenet_remote.py
- tools/prepare_shapenet_subset_manifest.py
- tools/download_shapenet_subset.py
- tools/preprocess_shapenet_subset.py
- tools/generate_dataset_card.py

Reference docs:
- README.md
- docs/VPS_SETUP.md
- progress.md

## 4) VPS Baseline Setup

On VPS (Ubuntu):

```bash
sudo apt update
sudo apt -y upgrade
sudo apt -y install git curl unzip tmux htop build-essential python3 python3-venv python3-pip
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone <your-repo-url> r4-mesh
cd r4-mesh
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Create env file:

```bash
cp .env.example .env
```

Edit .env and set token:
- HUGGINGFACE_TOKEN=your_token

GPU check:

```bash
nvidia-smi
```

## 5) Two Data Paths You Can Choose

You have two valid options for getting train-ready data on VPS.

### Option A: Download And Process Dataset On VPS

Use this when VPS has good internet and enough disk.

Step A1: Explore metadata (optional but recommended)

```bash
source .venv/bin/activate
python tools/explore_shapenet_remote.py --env-file .env
```

Step A2: Build deterministic subset manifest

```bash
python tools/prepare_shapenet_subset_manifest.py \
  --env-file .env \
  --bundle furniture_plus_display \
  --cap-per-class 1000 \
  --seed 42 \
  --manifest-out artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json \
  --paths-out artifacts/shapenet_subsets/furniture_plus_display_cap1000_paths.txt
```

Step A3: Download selected files only

```bash
python tools/download_shapenet_subset.py \
  --manifest artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json \
  --env-file .env \
  --output-dir data/shapenet_subsets/furniture_plus_display_cap1000 \
  --workers 8
```

Step A4: Preprocess and split dataset

```bash
python tools/preprocess_shapenet_subset.py \
  --input-root data/shapenet_subsets/furniture_plus_display_cap1000 \
  --output-root artifacts/shapenet_furniture_plus_display_cap1000_preprocessed \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42 --min-vertices 16 --min-faces 16
```

Step A5: Generate dataset card

```bash
python tools/generate_dataset_card.py \
  --dataset-name shapenet_furniture_plus_display_cap1000_preprocessed \
  --dataset-root artifacts/shapenet_furniture_plus_display_cap1000_preprocessed \
  --summary artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/reports/summary.json \
  --train artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/train.json \
  --val artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/val.json \
  --test artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/test.json \
  --zip artifacts/shapenet_furniture_plus_display_cap1000_preprocessed.zip \
  --output artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/dataset_card.json
```

After this, continue to Section 6.

### Option B: Upload Preprocessed Zip Directly To VPS

Use this when local machine already has preprocessed artifact and you want to skip VPS preprocessing.

Expected local artifact:
- artifacts/shapenet_furniture_plus_display_cap1000_preprocessed.zip

Step B1: Upload zip from local to VPS

From local machine:

```bash
scp -P <PORT> artifacts/shapenet_furniture_plus_display_cap1000_preprocessed.zip <user>@<host>:~/datasets/
```

Step B2: Extract on VPS

On VPS:

```bash
mkdir -p ~/datasets/run
unzip -oq ~/datasets/shapenet_furniture_plus_display_cap1000_preprocessed.zip -d ~/datasets/run
```

You should now have dataset root at one of:
- ~/datasets/run
- ~/datasets/run/shapenet_furniture_plus_display_cap1000_preprocessed

Step B3: Confirm manifest exists

```bash
find ~/datasets/run -name train.json
```

After this, continue to Section 6.

## 6) Where To Go From There (Training)

Start a tmux session first:

```bash
tmux new -s r4train
source .venv/bin/activate
```

Run training with per-step checkpointing:

```bash
python tools/train_offline_grpo_with_checkpoints.py \
  --train-manifest artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/train.json \
  --val-manifest artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/val.json \
  --test-manifest artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/test.json \
  --dataset-root artifacts/shapenet_furniture_plus_display_cap1000_preprocessed \
  --steps 20000 \
  --group-size 8 \
  --checkpoint-dir artifacts/checkpoints/offline_grpo \
  --save-every-steps 1 \
  --keep-last 200 \
  --eval-every-steps 100 \
  --max-eval-groups 16 \
  --run-name shapenet_exp01 \
  --experiment-dir artifacts/experiments/offline_grpo
```

If your extracted dataset path is outside repo (Option B), point to that absolute path instead:

```bash
python tools/train_offline_grpo_with_checkpoints.py \
  --train-manifest /home/<user>/datasets/run/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/train.json \
  --val-manifest /home/<user>/datasets/run/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/val.json \
  --test-manifest /home/<user>/datasets/run/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/test.json \
  --dataset-root /home/<user>/datasets/run/shapenet_furniture_plus_display_cap1000_preprocessed \
  --steps 20000 \
  --group-size 8 \
  --checkpoint-dir artifacts/checkpoints/offline_grpo \
  --save-every-steps 1 \
  --keep-last 200 \
  --eval-every-steps 100 \
  --max-eval-groups 16 \
  --run-name shapenet_exp01 \
  --experiment-dir artifacts/experiments/offline_grpo
```

Experiment logs produced per run:
- artifacts/experiments/offline_grpo/shapenet_exp01/run_info.json
- artifacts/experiments/offline_grpo/shapenet_exp01/train_timeline.jsonl
- artifacts/experiments/offline_grpo/shapenet_exp01/eval_timeline.jsonl
- artifacts/experiments/offline_grpo/shapenet_exp01/latest_metrics.json

Use these for charts of step vs reward/cost/lambda/objective and periodic val/test trends.

Detach tmux:
- Ctrl+B, then D

Reattach later:

```bash
tmux attach -t r4train
```

## 7) Resume Workflow (Same Machine Or New Machine)

Resume command:

```bash
python tools/train_offline_grpo_with_checkpoints.py \
  --steps 20000 \
  --group-size 8 \
  --checkpoint-dir artifacts/checkpoints/offline_grpo \
  --save-every-steps 1 \
  --eval-every-steps 100 \
  --max-eval-groups 16 \
  --run-name shapenet_exp01 \
  --experiment-dir artifacts/experiments/offline_grpo \
  --keep-last 200 \
  --resume
```

If switching machines, copy these first:
- artifacts/checkpoints/offline_grpo/
- preprocessed dataset root used in previous run
- matching manifest files
- same repo commit and dependencies

## 8) Quick Verification Checklist

Before training:
- .env exists and has HUGGINGFACE_TOKEN
- train manifest path is valid
- dataset root path is valid
- enough disk space for checkpoints and data

During training:
- checkpoints are appearing in artifacts/checkpoints/offline_grpo/
- latest pointer exists: artifacts/checkpoints/offline_grpo/latest.json
- train timeline grows: artifacts/experiments/offline_grpo/<run_name>/train_timeline.jsonl
- eval timeline grows: artifacts/experiments/offline_grpo/<run_name>/eval_timeline.jsonl

After interruptions:
- run with --resume and verify step number increases from previous latest

## 9) Notes

- This repo ignores large data and artifacts in git by design.
- .env must never be committed.
- For deeper operational and recovery details, also see docs/VPS_SETUP.md.
