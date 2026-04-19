#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json}"
ENV_FILE="${2:-.env}"
OUTPUT_DIR="${3:-data/shapenet_subsets/furniture_plus_display_cap1000}"
WORKERS="${4:-8}"
LIMIT="${5:-0}"

ARGS=(
  run
  python
  tools/download_shapenet_subset.py
  --manifest "$MANIFEST"
  --env-file "$ENV_FILE"
  --output-dir "$OUTPUT_DIR"
  --workers "$WORKERS"
)

if [[ "$LIMIT" != "0" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

echo "Running: uv ${ARGS[*]}"
uv "${ARGS[@]}"

echo "Subset download complete."
