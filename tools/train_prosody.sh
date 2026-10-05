#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WAV_DIR="${VOXERA_PROSODY_WAV_DIR:-${VOXERA_TRAIN_WAV_DIR:-$ROOT_DIR/data/train_wavs}}"
OUTPUT_DIR="${VOXERA_CHECKPOINT_DIR:-$ROOT_DIR/artifacts/checkpoints}"
EPOCHS="${VOXERA_PROSODY_EPOCHS:-8}"
ACCUM="${VOXERA_GRAD_ACCUM:-4}"
TRAIN_PYTHON="${VOXERA_TRAIN_PYTHON:-python}"

if [ ! -d "$WAV_DIR" ]; then
  echo "prosody WAV directory not found: $WAV_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
"$TRAIN_PYTHON" -m training.train_prosody \
  --wav-dir "$WAV_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --gradient-accumulation "$ACCUM"
