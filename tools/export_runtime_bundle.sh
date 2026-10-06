#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
CHECKPOINT_DIR="${VOXERA_CHECKPOINT_DIR:-$ROOT_DIR/artifacts/checkpoints}"
OUTPUT_DIR="${VOXERA_RUNTIME_BUNDLE_DIR:-$ROOT_DIR/artifacts/runtime_bundle}"
TRAIN_PYTHON="${VOXERA_TRAIN_PYTHON:-python}"
CHECKPOINT="${VOXERA_JOINT_CHECKPOINT:-}"

if [ -z "$CHECKPOINT" ]; then
  CHECKPOINT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f \
    -name 'joint-refinement-epoch*.pt' -print 2>/dev/null \
    | sort | tail -n 1)
fi

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
  echo "No joint-refinement checkpoint found." >&2
  echo "Expected one below: $CHECKPOINT_DIR" >&2
  echo "Run sh tools/train_m1.sh after training data/GPU are available." >&2
  exit 1
fi

cd "$ROOT_DIR"
"$TRAIN_PYTHON" -m training.export_bundle \
  --joint-checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR"

echo
echo "runtime_bundle=$OUTPUT_DIR"
echo "source_checkpoint=$CHECKPOINT"
echo "manifest=$OUTPUT_DIR/bundle.json"
