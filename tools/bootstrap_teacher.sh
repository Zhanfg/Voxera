#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TEACHER_ROOT="$ROOT_DIR/.cache/teacher"
MEANVC2_DIR="$TEACHER_ROOT/MeanVC2"
MEANVC2_REPO="https://github.com/ASLP-lab/MeanVC2.git"
MEANVC2_COMMIT="13acf84c1bf135ea5edad9c245b345289b06b33e"
TEACHER_PYTHON="${VOXERA_TEACHER_PYTHON:-python}"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require git
require "$TEACHER_PYTHON"

mkdir -p "$TEACHER_ROOT"

if [ ! -d "$MEANVC2_DIR/.git" ]; then
  echo "[1/4] cloning pinned MeanVC2 teacher"
  git init "$MEANVC2_DIR"
  git -C "$MEANVC2_DIR" remote add origin "$MEANVC2_REPO"
fi

echo "[2/4] fetching teacher commit $MEANVC2_COMMIT"
git -C "$MEANVC2_DIR" fetch --depth 1 origin "$MEANVC2_COMMIT"
git -C "$MEANVC2_DIR" checkout --detach FETCH_HEAD

ACTUAL_COMMIT=$(git -C "$MEANVC2_DIR" rev-parse HEAD)
if [ "$ACTUAL_COMMIT" != "$MEANVC2_COMMIT" ]; then
  echo "teacher commit mismatch: $ACTUAL_COMMIT" >&2
  exit 1
fi
printf '%s\n' "$ACTUAL_COMMIT" > "$TEACHER_ROOT/meanvc2.commit"

echo "[3/4] checking the full preprocessing environment"
if ! MEANVC2_DIR="$MEANVC2_DIR" "$TEACHER_PYTHON" - <<'PY'
import os
import sys

teacher = os.environ["MEANVC2_DIR"]
sys.path.insert(0, teacher)

modules = [
    "numpy",
    "torch",
    "torchaudio",
    "soundfile",
    "tqdm",
    "librosa",
    "soxr",
    "huggingface_hub",
]
missing = []
for module in modules:
    try:
        __import__(module)
    except Exception:
        missing.append(module)

if missing:
    raise SystemExit("missing Python modules: " + ", ".join(missing))

from torchaudio.compliance import kaldi  # noqa: F401
from preprocess.audio import load_wav, melspectrogram  # noqa: F401
from preprocess.models.ecapa_tdnn import ECAPA_TDNN_SMALL  # noqa: F401
PY
then
  echo
  echo "MeanVC2 source is pinned, but its preprocessing environment is incomplete."
  echo "Set VOXERA_TEACHER_PYTHON to a dedicated Python 3.11 interpreter with"
  echo "the upstream MeanVC2 preprocessing dependencies installed."
  exit 2
fi

echo "[4/4] downloading official MeanVC2 preprocessing checkpoints"
(
  cd "$MEANVC2_DIR"
  "$TEACHER_PYTHON" initialization.py --task preprocess
)

for checkpoint in \
  "$MEANVC2_DIR/preprocess/ckpts/fastu2pp_80ms.pt" \
  "$MEANVC2_DIR/preprocess/ckpts/wavlm_large_finetune.pth"
do
  if [ ! -s "$checkpoint" ]; then
    echo "teacher checkpoint missing after initialization: $checkpoint" >&2
    exit 1
  fi
done

echo
echo "teacher_python=$TEACHER_PYTHON"
echo "teacher_source=$MEANVC2_DIR"
echo "teacher_commit=$ACTUAL_COMMIT"
echo "teacher checkpoints ready"
