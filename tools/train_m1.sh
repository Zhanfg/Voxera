#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WAV_DIR="${VOXERA_TRAIN_WAV_DIR:-$ROOT_DIR/data/train_wavs}"
CACHE_DIR="${VOXERA_TEACHER_CACHE_DIR:-$ROOT_DIR/artifacts/teacher_cache}"
MANIFEST="${VOXERA_TEACHER_MANIFEST:-$CACHE_DIR/manifest.jsonl}"
CHECKPOINT_DIR="${VOXERA_CHECKPOINT_DIR:-$ROOT_DIR/artifacts/checkpoints}"
TEACHER_PYTHON="${VOXERA_TEACHER_PYTHON:-python}"
TRAIN_PYTHON="${VOXERA_TRAIN_PYTHON:-python}"
SKIP_BOOTSTRAP="${VOXERA_SKIP_TEACHER_BOOTSTRAP:-0}"

export VOXERA_TRAIN_WAV_DIR="$WAV_DIR"
export VOXERA_TEACHER_CACHE_DIR="$CACHE_DIR"
export VOXERA_TEACHER_MANIFEST="$MANIFEST"
export VOXERA_CHECKPOINT_DIR="$CHECKPOINT_DIR"
export VOXERA_TEACHER_PYTHON="$TEACHER_PYTHON"
export VOXERA_TRAIN_PYTHON="$TRAIN_PYTHON"

echo "Voxera M1 training"
echo "root=$ROOT_DIR"
echo "wav_dir=$WAV_DIR"
echo "teacher_cache=$CACHE_DIR"
echo "checkpoint_dir=$CHECKPOINT_DIR"
echo "teacher_python=$TEACHER_PYTHON"
echo "train_python=$TRAIN_PYTHON"
echo

if [ ! -d "$WAV_DIR" ]; then
  echo "training WAV directory not found: $WAV_DIR" >&2
  echo "Place flat .wav files there or set VOXERA_TRAIN_WAV_DIR." >&2
  exit 1
fi

echo "[1/6] validating Voxera training environment and WAV inputs"
(
  cd "$ROOT_DIR"
  "$TRAIN_PYTHON" - <<'PY'
import torch
import voxera

print(f"voxera={voxera.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
  "$TRAIN_PYTHON" -m training.audit_wavs --wav-dir "$WAV_DIR"
)

if [ "$SKIP_BOOTSTRAP" = "1" ]; then
  echo "[2/6] teacher bootstrap skipped by VOXERA_SKIP_TEACHER_BOOTSTRAP=1"
else
  echo "[2/6] bootstrapping pinned MeanVC2 teacher"
  sh "$ROOT_DIR/tools/bootstrap_teacher.sh"
fi

echo "[3/6] preparing fresh teacher cache"
sh "$ROOT_DIR/tools/prepare_teacher_cache.sh"

echo "[4/6] training ContentNet + TimbreNet"
sh "$ROOT_DIR/tools/train_encoders.sh"

echo "[5/6] training ConditionFusion + DecoderNet + LiteVocoder"
sh "$ROOT_DIR/tools/train_generators.sh"

echo "[6/6] jointly refining the full five-model graph"
sh "$ROOT_DIR/tools/train_joint.sh"

echo
echo "M1 training pipeline complete"
echo "manifest=$MANIFEST"
echo "checkpoints=$CHECKPOINT_DIR"
