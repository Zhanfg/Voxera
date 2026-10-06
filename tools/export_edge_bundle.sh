#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CHECKPOINT_DIR="${VOXERA_CHECKPOINT_DIR:-$ROOT_DIR/artifacts/checkpoints}"
OUTPUT_DIR="${VOXERA_EDGE_BUNDLE_DIR:-$ROOT_DIR/artifacts/edge_bundle}"
TRAIN_PYTHON="${VOXERA_TRAIN_PYTHON:-python}"
ALLOW_UNTRAINED="${VOXERA_ALLOW_UNTRAINED_EXPORT:-0}"

cd "$ROOT_DIR"
if [ "$ALLOW_UNTRAINED" = "1" ]; then
  "$TRAIN_PYTHON" -m training.export_bundle \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --allow-untrained
else
  "$TRAIN_PYTHON" -m training.export_bundle \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR"
fi

echo
echo "edge_bundle=$OUTPUT_DIR"
echo "manifest=$OUTPUT_DIR/manifest.json"
