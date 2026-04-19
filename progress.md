# Progress Log

## 2026-04-17 - Session 1

- Decision: Start with dataset-independent engineering work while ShapeNet access is pending.
- Decision: Build a local offline scaffold for RLVR components first (reward checks, GRPO math, Lagrangian controller, synthetic smoke tests).
- Rationale: This maximizes local machine time and reduces GPU-hour waste later on VPS.
- Next step: Implement reusable Python modules and a smoke test using synthetic meshes.

## 2026-04-17 - Session 1 Milestone Update

- Progress: Created a local Python scaffold for dataset-independent RLVR work.
- Added: [requirements.txt](requirements.txt) with core dependencies needed for reward checks and local testing.
- Added: [src/r4/grpo.py](src/r4/grpo.py) with GRPO group-relative advantage normalization.
- Added: [src/r4/lagrangian.py](src/r4/lagrangian.py) with a PI controller for Lagrangian penalty updates.
- Added: [src/r4/reward_engine.py](src/r4/reward_engine.py) implementing:
	- Trimesh-based watertightness check.
	- Optional Open3D self-intersection check.
	- Boundary Edge Ratio metric.
	- Triangle quality metric proxy for topology quality.
	- Combined reward and cost output suitable for GRPO training loops.
- Added: [tools/smoke_reward_checks.py](tools/smoke_reward_checks.py) to validate reward logic with synthetic meshes (closed box and open box).
- Decision: Keep current implementation lightweight and deterministic so development can continue fully offline until dataset approval and VPS GPU access.
- Validation: Static error check reports no issues in all newly added Python files.

## 2026-04-17 - Session 2 Start

- Decision: Use terminal-driven local virtual environment setup with uv for package installation and verification runs.
- Rationale: Keeps dependency state isolated and reproducible before VPS GPU execution.
- Progress: Attempted `uv pip install` in local `.venv` and identified a blocker: Open3D wheels are unavailable for Python 3.13 (cp313).
- Decision: Keep Open3D optional for local cp313 development by adding a Python version marker in [requirements.txt](requirements.txt), while preserving reward-engine fallback behavior when Open3D is absent.

## 2026-04-17 - Session 2 Milestone Update

- Progress: Recreated local virtual environment using terminal and uv (`uv venv --clear .venv`) and installed dependencies from [requirements.txt](requirements.txt).
- Added: [src/r4/data.py](src/r4/data.py) with dataset-ready utilities:
	- Vertex normalization to bounded coordinate range.
	- Deterministic Z/Y/X vertex sorting with face index remapping.
	- Mesh loading helpers and ShapeNet synset file discovery.
	- Manifest sampling and JSON manifest writing.
- Added: [src/r4/trainer.py](src/r4/trainer.py) with offline GRPO scaffold:
	- Lagrangian group objective helper.
	- Group evaluation pipeline producing rewards, costs, advantages, and lambda updates.
	- Resampling signal for low-variance reward groups.
- Added: [tools/prepare_shapenet_manifest.py](tools/prepare_shapenet_manifest.py) for post-approval dataset indexing and sampling.
- Added: test suite under [tests](tests):
	- [tests/test_reward_engine.py](tests/test_reward_engine.py)
	- [tests/test_grpo.py](tests/test_grpo.py)
	- [tests/test_lagrangian.py](tests/test_lagrangian.py)
	- [tests/test_data.py](tests/test_data.py)
	- [tests/conftest.py](tests/conftest.py) for source-path setup.
- Validation: Ran `uv run --python .\\.venv\\Scripts\\python.exe -m pytest -q` and all tests passed (10/10).
- Validation: Ran synthetic runtime check via [tools/smoke_reward_checks.py](tools/smoke_reward_checks.py); confirmed expected behavior:
	- Closed mesh receives boolean pass and high reward.
	- Open mesh receives boolean fail and non-zero constraint cost.
	- GRPO advantage normalization produces centered per-group scores.
	- PI controller updates lambda upward when observed cost remains above target.

## 2026-04-17 - Session 3 Start

- Decision: Process downloaded Google Scanned Objects locally (preprocessing + partitioning) and prepare a VPS-ready zip artifact.
- Rationale: Offloads all CPU and data-engineering work from constrained VPS GPU hours.
- Progress: Located dataset at [Object Scanning](Object%20Scanning) and profiled contents.
- Observation: Current download contains 73 `.obj` meshes (no `.glb/.ply/.off/.stl/.fbx` detected).
- Decision: Build and run a local preprocessing pipeline for OBJ-first ingestion, normalization, cleaning, and stratified partitioning by top-level category folder.

## 2026-04-17 - Session 3 Run Verification

- Progress: Verified completed preprocessing run and artifact generation at [artifacts/gso_preprocessed](artifacts/gso_preprocessed) and [artifacts/gso_preprocessed.zip](artifacts/gso_preprocessed.zip).
- Result snapshot: discovered=73, processed=66, rejected=7, splits train/val/test=42/12/12.
- Observation: All 7 rejections were due to missing Pillow dependency (`No module named 'PIL'`) during OBJ handling for texture-linked assets.
- Decision: Add Pillow to [requirements.txt](requirements.txt), install with uv, and rerun preprocessing to maximize retained samples.
- Progress: Confirmed the completed run artifacts and counts in [artifacts/gso_preprocessed/reports/summary.json](artifacts/gso_preprocessed/reports/summary.json) and [artifacts/gso_preprocessed.zip](artifacts/gso_preprocessed.zip).
- Cleanup: Stopped the additional background rerun terminal to avoid unnecessary CPU usage while keeping verified completed-run artifacts intact.

## 2026-04-17 - Session 3 Finalization (All 3 Requested Actions)

- Action 1 complete: Regenerated dataset after ensuring Pillow availability and performing a clean rebuild.
- Result: [artifacts/gso_preprocessed/reports/summary.json](artifacts/gso_preprocessed/reports/summary.json) now reports discovered=73, processed=73, rejected=0.
- Split result: train=47, val=13, test=13.

- Action 2 complete: Added one-shot VPS transfer and extraction script at [tools/upload_gso_to_vps.ps1](tools/upload_gso_to_vps.ps1).
- Purpose: Creates remote directory, uploads zip via scp, extracts on VPS via ssh+unzip, and prints remote directory overview.

- Action 3 complete: Added compact dataset card generator [tools/generate_dataset_card.py](tools/generate_dataset_card.py) and generated [artifacts/gso_preprocessed/dataset_card.json](artifacts/gso_preprocessed/dataset_card.json).
- Dataset card includes source counts, split counts/ratios, category distributions per split, zip size, and zip SHA256.

- Final transfer artifact: [artifacts/gso_preprocessed.zip](artifacts/gso_preprocessed.zip) (90,875,920 bytes).

## 2026-04-17 - Dataset Adequacy Assessment

- Decision: Current local Object Scanning dataset is sufficient for pipeline validation and small proof-of-concept RLVR experiments, but not sufficient as the sole dataset for robust final alignment claims.
- Evidence from [artifacts/gso_preprocessed/manifests/all.json](artifacts/gso_preprocessed/manifests/all.json):
	- Total samples: 73 across 8 categories.
	- Unique source object directories: 43 (multiple OBJ files per object in several cases).
	- Strong class imbalance (e.g., Furniture 25 vs Statuettes-Figurines 2).
	- All 73 meshes currently non-watertight in baseline metadata.
- Implication: Keep this dataset for local preprocessing/debug/sanity benchmarks; use larger approved dataset on VPS (or merged corpus) for final training and evaluation.

## 2026-04-19 - ShapeNet Remote Exploration (No Bulk Download)

- Progress: Verified gated-access read for `ShapeNet/shapenetcore-glb` using token from [.env](.env) (key detected as `HUGGINGFACE_TOKEN`).
- Discovery summary (metadata only):
	- Repository reports 51,486 `.glb` files, 52 active class folders in GLB layout, and total size ~63.43 GB.
	- Path schema is flat per class (`class/hash.glb`), so file discovery and selective sync are straightforward.
	- Dataset card/README pulled only (no mesh payload download).
- Added reusable exploration utility: [tools/explore_shapenet_remote.py](tools/explore_shapenet_remote.py).
- Generated planning artifacts:
	- [artifacts/shapenet_exploration.json](artifacts/shapenet_exploration.json)
	- [artifacts/shapenet_subset_options.json](artifacts/shapenet_subset_options.json)
- Decision: Postpone full download until subset is selected; use generated size/count estimates to choose a cap-per-class strategy first.

## 2026-04-19 - ShapeNet Subset Selection Finalized

- Decision: Selected first recommended subset option: `furniture_plus_display` with cap `1000` per class.
- Added manifest builder: [tools/prepare_shapenet_subset_manifest.py](tools/prepare_shapenet_subset_manifest.py).
- Added selective downloader: [tools/download_shapenet_subset.py](tools/download_shapenet_subset.py).
- Added convenience wrappers:
	- Windows: [tools/download_furniture_plus_display_cap1000.ps1](tools/download_furniture_plus_display_cap1000.ps1)
	- Linux/VPS: [tools/download_furniture_plus_display_cap1000.sh](tools/download_furniture_plus_display_cap1000.sh)

- Generated selected subset artifacts:
	- [artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json](artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json)
	- [artifacts/shapenet_subsets/furniture_plus_display_cap1000_paths.txt](artifacts/shapenet_subsets/furniture_plus_display_cap1000_paths.txt)

- Final selected subset summary (metadata-derived, no bulk download):
	- Files: 8,685
	- Size estimate from exact selected records: ~6.045 GB
	- Source repo: `ShapeNet/shapenetcore-glb`

- Validation: ran probe selective download with `--limit 5`; download flow succeeded with 0 failures.
- Progress: Started full selective download for the chosen subset (8,685 files) into `data/shapenet_subsets/furniture_plus_display_cap1000` using manifest-driven downloader.

## 2026-04-19 - Readiness Check (Subset + Training)

- Status snapshot: selected subset download is in progress (expected 8,685 files; currently downloaded 873; remaining 7,812).
- Decision: Chosen subset size is sufficient for the next phase (subset preprocessing + training-pipeline bring-up), but not yet sufficient for full execution until download finishes.
- Validation completed:
	- Unit/integration tests currently pass (`13 passed`).
	- Download pipeline probe run succeeded earlier (`--limit 5`, 0 failures).
- Not yet validated end-to-end:
	- No full RL training run on actual ShapeNet subset yet.
	- No MeshGPT weight-loading/inference-train cycle validation yet.
	- No final ShapeNet-subset zip artifact generated yet (only GSO zip exists currently).

## 2026-04-19 - ShapeNet Subset Download Completion

- Progress: Manifest-driven subset download completed successfully for `furniture_plus_display_cap1000`.
- Result: downloaded file count matches manifest selection count (8,685) with no failure report.
- Next step: preprocess/split/package this subset into a VPS-ready training artifact.

## 2026-04-19 - ShapeNet Subset Preprocessing + Packaging Complete

- Progress: Completed preprocessing pipeline on downloaded subset via [tools/preprocess_shapenet_subset.py](tools/preprocess_shapenet_subset.py).
- Output root: [artifacts/shapenet_furniture_plus_display_cap1000_preprocessed](artifacts/shapenet_furniture_plus_display_cap1000_preprocessed)
- Output zip: [artifacts/shapenet_furniture_plus_display_cap1000_preprocessed.zip](artifacts/shapenet_furniture_plus_display_cap1000_preprocessed.zip)

- Summary from [artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/reports/summary.json](artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/reports/summary.json):
	- Discovered: 8,685
	- Processed: 7,453
	- Rejected: 1,232
	- Splits: train=5,217, val=1,119, test=1,117
	- Rejection rate: 14.19%

- Generated dataset card: [artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/dataset_card.json](artifacts/shapenet_furniture_plus_display_cap1000_preprocessed/dataset_card.json)
	- Zip SHA256: `14d9df76b861fbc463d7ffd17b803595136216b9ccadf4bf1dc13245793d9b8b`
	- Zip size: 530,457,593 bytes

- Validation updates:
	- Test suite remains green (`13 passed`).
	- Real-data trainer smoke check succeeded using 8 preprocessed meshes with [src/r4/trainer.py](src/r4/trainer.py) `OfflineGRPOTrainer.evaluate_group`.

- Remaining gap explicitly noted:
	- Full MeshGPT policy loading + RL fine-tuning on GPU VPS has not yet been executed.

## 2026-04-19 - Step-Level Checkpointing + Resume Support Implemented

- Decision: Add checkpointing at every training step with resume-from-latest semantics to tolerate free-tier machine switches without losing progress.
- Added trainer serialization support in [src/r4/trainer.py](src/r4/trainer.py):
	- `global_step` tracking persisted in trainer state.
	- `state_dict()` for serializing train config and controller internals.
	- `load_state_dict()` for restoring controller PI state (`lambda`, gains, bounds, target, and integral accumulator).

- Added checkpoint manager in [src/r4/checkpointing.py](src/r4/checkpointing.py):
	- `save_training_checkpoint(...)` writes atomic step checkpoints and updates `latest.json` pointer.
	- `load_training_checkpoint(...)` and `load_latest_training_checkpoint(...)` restore trainer and optional model/optimizer/scheduler states.
	- Retention policy support via `keep_last_n` checkpoint pruning.

- Added runnable training loop with checkpoint/resume integration in [tools/train_offline_grpo_with_checkpoints.py](tools/train_offline_grpo_with_checkpoints.py):
	- Supports `--save-every-steps` (default `1`) for per-step saves.
	- Supports `--resume` to continue from latest checkpoint in a given directory.
	- Emits run summary metrics to JSON for machine-to-machine handoff visibility.

- Export update in [src/r4/__init__.py](src/r4/__init__.py): checkpointing utilities exposed in package API.

- Added tests in [tests/test_checkpointing.py](tests/test_checkpointing.py):
	- Round-trip save/load validation for trainer/controller state.
	- Old-checkpoint pruning validation.

- Validation:
	- Full test suite passed locally: `15 passed`.
	- New checkpoint tests passed: `2 passed`.
	- Resume smoke run validated on preprocessed ShapeNet subset:
		- First run ended at step 3.
		- Resume run started from step 3 and ended at step 5.
		- Checkpoint directory contains `latest.json` and `step_00000001..05.ckpt`.

## 2026-04-19 - Repository Initialization + Git Hygiene + VPS Documentation

- Decision: Initialize this workspace as a git repository now that core scaffolding, data tooling, and checkpoint-resume support are in place.
- Progress: Created project-level [.gitignore](.gitignore) with explicit coverage for:
	- secrets (`.env`, `.env.*` with `.env.example` allowed)
	- local Python env/cache noise (`.venv`, `__pycache__`, pytest/mypy/ruff caches)
	- large generated/data paths (`data/`, `artifacts/`, `Object Scanning/`)
	- common large binary outputs (`*.zip`, `*.npz`, `*.ckpt`, etc.)
- Validation: Ran git initialization (`git init -b main`) and status check; only source/docs/config files are visible for tracking, while large dataset/artifact paths are excluded.

- Added [.env.example](.env.example) as a safe template for token configuration.

- Added detailed repository documentation in [README.md](README.md):
	- project scope and module map
	- deterministic ShapeNet subset workflow (explore -> manifest -> selective download -> preprocess -> dataset card)
	- checkpointed offline training commands and resume behavior
	- quick health checks and operational notes

- Added dedicated VPS runbook in [docs/VPS_SETUP.md](docs/VPS_SETUP.md):
	- provisioning and system bootstrap
	- GPU/runtime validation
	- data transfer strategies (local preprocess upload vs on-VPS download/preprocess)
	- tmux-based long-run execution
	- same-machine and cross-machine checkpoint resume procedures
	- failure recovery playbooks and security checklist
