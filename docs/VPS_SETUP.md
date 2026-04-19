# VPS Setup and Operations Guide (GPU + Resume-Safe Training)

This guide is designed for running long jobs on preemptible/free-tier VPS environments while minimizing lost progress.

## Goals

- Keep GPU time focused on training, not preprocessing
- Make training resumable from the latest step checkpoint
- Support handoff across machines with deterministic state

## Reference Topology

- Local workstation:
  - Data exploration/subset planning
  - Optional full preprocessing
  - Checkpoint backup origin/target
- VPS (GPU):
  - Training execution
  - Checkpoint writing
  - Optional dataset download/preprocess (if local transfer is not practical)

## 1) Provision a GPU VPS

Recommended baseline:
- Ubuntu 22.04 LTS or newer
- NVIDIA L4-class GPU (or equivalent)
- >= 100 GB disk for datasets, checkpoints, and temporary archives
- Stable outbound internet (for Hugging Face access)

Create a non-root sudo user and set SSH key auth before starting workloads.

## 2) Base System Bootstrap (Ubuntu)

```bash
sudo apt update
sudo apt -y upgrade
sudo apt -y install git curl unzip tmux htop build-essential python3 python3-venv python3-pip
```

Install uv (choose one method):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# re-open shell or source profile so `uv` is on PATH
```

## 3) NVIDIA / CUDA Validation

Run:
```bash
nvidia-smi
```

Expected outcome:
- GPU visible
- driver loaded
- no persistent hardware errors

Optional PyTorch CUDA quick check after env setup:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## 4) Clone Repo and Prepare Environment

```bash
git clone <your-repo-url> r4-mesh
cd r4-mesh
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Create `.env`:
```bash
cp .env.example .env
# edit and set token
nano .env
```

Required token key:
- `HUGGINGFACE_TOKEN` (or `HUGGING_FACE_TOKEN` / `HF_TOKEN`)

## 5) Choose Data Strategy

### Strategy A: Preprocess Locally, Upload Artifacts (preferred when local CPU is stronger)

What to transfer to VPS:
- zipped preprocessed dataset artifact
- dataset card/manifests/reports

Transfer command pattern:
```bash
scp -P <PORT> <local_zip> <user>@<host>:~/datasets/
ssh -p <PORT> <user>@<host> "mkdir -p ~/datasets/run && unzip -oq ~/datasets/<zip_name> -d ~/datasets/run"
```

### Strategy B: Download + Preprocess on VPS (preferred when network proximity to HF is strong)

1. Build or copy manifest JSON to VPS.
2. Download only listed files:
```bash
source .venv/bin/activate
python tools/download_shapenet_subset.py \
  --manifest artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json \
  --env-file .env \
  --output-dir data/shapenet_subsets/furniture_plus_display_cap1000 \
  --workers 8
```
3. Preprocess:
```bash
python tools/preprocess_shapenet_subset.py \
  --input-root data/shapenet_subsets/furniture_plus_display_cap1000 \
  --output-root artifacts/shapenet_furniture_plus_display_cap1000_preprocessed \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42 --min-vertices 16 --min-faces 16
```

## 6) Long-Running Training with tmux

Start a persistent session:
```bash
tmux new -s r4train
source .venv/bin/activate
```

Run training with per-step checkpointing:
```bash
python tools/train_offline_grpo_with_checkpoints.py \
  --train-manifest artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/manifests/train.json \
  --dataset-root artifacts/shapenet_furniture_plus_display_cap1000_preprocessed \
  --steps 20000 \
  --group-size 8 \
  --checkpoint-dir artifacts/checkpoints/offline_grpo \
  --save-every-steps 1 \
  --keep-last 200
```

Detach session:
- `Ctrl+B`, then `D`

Reattach later:
```bash
tmux attach -t r4train
```

## 7) Resume After Preemption or Machine Switch

### Same machine resume

```bash
source .venv/bin/activate
python tools/train_offline_grpo_with_checkpoints.py \
  --steps 20000 \
  --group-size 8 \
  --checkpoint-dir artifacts/checkpoints/offline_grpo \
  --save-every-steps 1 \
  --keep-last 200 \
  --resume
```

### Different machine resume

Copy these from old machine to new machine:
- checkpoint directory (contains `latest.json` + `step_*.ckpt`)
- the exact preprocessed dataset root used in training
- matching train manifest JSON
- same repo commit and Python dependencies

Transfer example:
```bash
# old machine -> local or directly to new machine
rsync -avz artifacts/checkpoints/offline_grpo/ <user>@<new-host>:~/r4-mesh/artifacts/checkpoints/offline_grpo/
rsync -avz artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/ <user>@<new-host>:~/r4-mesh/artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/
```

Resume command on new machine should use same paths/parameters.

## 8) Operational Hygiene

- Keep checkpoints on a dedicated disk path if ephemeral root disks are likely.
- Sync checkpoints off-machine periodically (e.g., every 5-15 minutes).
- Monitor free disk space:
```bash
df -h
```
- Monitor GPU and process health:
```bash
watch -n 2 nvidia-smi
```

## 9) Recovery Playbooks

### Case A: Job crashed, machine still alive

1. Inspect last logs.
2. Verify checkpoint pointer:
```bash
cat artifacts/checkpoints/offline_grpo/latest.json
```
3. Resume with `--resume`.

### Case B: Machine terminated

1. Provision replacement VPS.
2. Restore repo at same commit.
3. Restore data + checkpoint directories.
4. Activate env, reinstall requirements.
5. Resume with same command and `--resume`.

### Case C: Corrupted newest checkpoint file

1. Inspect `step_*.ckpt` files.
2. Temporarily move the latest corrupted checkpoint.
3. Update `latest.json` or let loader fall back to lexicographically latest valid step file.
4. Resume.

## 10) Security Checklist

- `.env` must remain untracked in git.
- Use SSH keys, disable password login where possible.
- Restrict inbound ports to SSH only.
- Rotate Hugging Face token if exposure is suspected.

## 11) Recommended Run Checklist

Before starting a long run:
1. `pytest` passes on target environment.
2. Dataset manifest and root paths are correct.
3. `latest.json` is absent (fresh run) or verified (resume run).
4. Disk free space is sufficient for checkpoints and logs.
5. tmux session is active.

During run:
1. Confirm checkpoint files are being updated every step interval.
2. Tail metrics output file:
```bash
cat artifacts/checkpoints/offline_grpo/latest_metrics.json
```
3. Sync checkpoints out-of-band if VPS is preemptible.

After run:
1. Archive final checkpoint set.
2. Export metrics and run config for reproducibility.
3. Record run summary in project progress log.
