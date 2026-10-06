#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CHECKPOINT_DIR="${VOXERA_CHECKPOINT_DIR:-$ROOT_DIR/artifacts/checkpoints}"
OUTPUT_DIR="${VOXERA_EDGE_BUNDLE_DIR:-$ROOT_DIR/artifacts/edge_bundle}"
TRAIN_PYTHON="${VOXERA_TRAIN_PYTHON:-python}"

cd "$ROOT_DIR"
"$TRAIN_PYTHON" -m training.export_bundle \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$OUTPUT_DIR"

echo
echo "edge_bundle=$OUTPUT_DIR"
echo "manifest=$OUTPUT_DIR/manifest.json"
