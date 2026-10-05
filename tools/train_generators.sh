#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MANIFEST="${VOXERA_TEACHER_MANIFEST:-$ROOT_DIR/artifacts/teacher_cache/manifest.jsonl}"
OUTPUT_DIR="${VOXERA_CHECKPOINT_DIR:-$ROOT_DIR/artifacts/checkpoints}"
EPOCHS="${VOXERA_GENERATOR_EPOCHS:-10}"
ACCUM="${VOXERA_GRAD_ACCUM:-4}"
TRAIN_PYTHON="${VOXERA_TRAIN_PYTHON:-python}"

if [ ! -f "$MANIFEST" ]; then
  echo "teacher manifest not found: $MANIFEST" >&2
  echo "Run sh tools/prepare_teacher_cache.sh first." >&2
  exit 1
fi

cd "$ROOT_DIR"
"$TRAIN_PYTHON" -m training.train_generators \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --component both \
  --epochs "$EPOCHS" \
  --gradient-accumulation "$ACCUM"
