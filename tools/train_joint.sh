#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MANIFEST="${VOXERA_TEACHER_MANIFEST:-$ROOT_DIR/artifacts/teacher_cache/manifest.jsonl}"
CHECKPOINT_DIR="${VOXERA_CHECKPOINT_DIR:-$ROOT_DIR/artifacts/checkpoints}"
OUTPUT_DIR="${VOXERA_JOINT_CHECKPOINT_DIR:-$CHECKPOINT_DIR}"
EPOCHS="${VOXERA_JOINT_EPOCHS:-5}"
ACCUM="${VOXERA_GRAD_ACCUM:-2}"

if [ ! -f "$MANIFEST" ]; then
  echo "teacher manifest not found: $MANIFEST" >&2
  exit 1
fi

cd "$ROOT_DIR"
python -m training.train_joint \
  --manifest "$MANIFEST" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --gradient-accumulation "$ACCUM"
