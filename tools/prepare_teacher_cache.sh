#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WAV_DIR="${VOXERA_TRAIN_WAV_DIR:-$ROOT_DIR/data/train_wavs}"
CACHE_DIR="${VOXERA_TEACHER_CACHE_DIR:-$ROOT_DIR/artifacts/teacher_cache}"
TEACHER_DIR="$ROOT_DIR/.cache/teacher/MeanVC2"
PIN_FILE="$ROOT_DIR/.cache/teacher/meanvc2.commit"
EXPECTED_COMMIT="13acf84c1bf135ea5edad9c245b345289b06b33e"

if [ ! -d "$WAV_DIR" ]; then
  echo "training WAV directory not found: $WAV_DIR" >&2
  echo "Place flat .wav training files there, or set VOXERA_TRAIN_WAV_DIR." >&2
  exit 1
fi

if [ ! -d "$TEACHER_DIR/.git" ] || [ ! -f "$PIN_FILE" ]; then
  echo "teacher source is not bootstrapped; run:" >&2
  echo "  sh tools/bootstrap_teacher.sh" >&2
  exit 1
fi

ACTUAL_COMMIT=$(git -C "$TEACHER_DIR" rev-parse HEAD)
if [ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]; then
  echo "teacher checkout is not pinned to $EXPECTED_COMMIT" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR/bn" "$CACHE_DIR/mel" "$CACHE_DIR/speaker"

DEVICE=cpu
if python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  DEVICE=cuda
fi

echo "[1/4] extracting 40 ms / 256-d MeanVC2 bottlenecks"
python "$TEACHER_DIR/preprocess/extract_bn_80ms.py" \
  --input_dir "$WAV_DIR" \
  --output_dir "$CACHE_DIR/bn" \
  --device "$DEVICE"

echo "[2/4] extracting 10 ms / 80-bin MeanVC2 mel targets"
python "$TEACHER_DIR/preprocess/extract_mel.py" \
  --input_dir "$WAV_DIR" \
  --output_dir "$CACHE_DIR/mel" \
  --config preConfiged16K_10ms

echo "[3/4] extracting 256-d WavLM + ECAPA speaker embeddings"
python "$TEACHER_DIR/preprocess/extract_spk_emb.py" \
  --input_dir "$WAV_DIR" \
  --output_dir "$CACHE_DIR/speaker" \
  --device "$DEVICE"

echo "[4/4] validating cache and writing provenance manifest"
python "$ROOT_DIR/training/teacher_cache.py" \
  --wav-dir "$WAV_DIR" \
  --cache-dir "$CACHE_DIR"

echo
echo "teacher_cache=$CACHE_DIR"
echo "device=$DEVICE"
echo "manifest=$CACHE_DIR/manifest.jsonl"
