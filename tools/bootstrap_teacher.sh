#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TEACHER_ROOT="$ROOT_DIR/.cache/teacher"
MEANVC2_DIR="$TEACHER_ROOT/MeanVC2"
MEANVC2_REPO="https://github.com/ASLP-lab/MeanVC2.git"
MEANVC2_COMMIT="13acf84c1bf135ea5edad9c245b345289b06b33e"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require git
require python

mkdir -p "$TEACHER_ROOT"

if [ ! -d "$MEANVC2_DIR/.git" ]; then
  echo "[1/3] cloning pinned MeanVC2 teacher"
  git init "$MEANVC2_DIR"
  git -C "$MEANVC2_DIR" remote add origin "$MEANVC2_REPO"
fi

echo "[2/3] fetching teacher commit $MEANVC2_COMMIT"
git -C "$MEANVC2_DIR" fetch --depth 1 origin "$MEANVC2_COMMIT"
git -C "$MEANVC2_DIR" checkout --detach FETCH_HEAD

ACTUAL_COMMIT=$(git -C "$MEANVC2_DIR" rev-parse HEAD)
if [ "$ACTUAL_COMMIT" != "$MEANVC2_COMMIT" ]; then
  echo "teacher commit mismatch: $ACTUAL_COMMIT" >&2
  exit 1
fi
printf '%s\n' "$ACTUAL_COMMIT" > "$TEACHER_ROOT/meanvc2.commit"

echo "[3/3] checking teacher Python environment"
if ! python - <<'PY'
modules = ["numpy", "torch", "torchaudio", "soundfile", "tqdm"]
missing = []
for module in modules:
    try:
        __import__(module)
    except Exception:
        missing.append(module)
if missing:
    raise SystemExit("missing Python modules: " + ", ".join(missing))
PY
then
  echo
  echo "MeanVC2 source is pinned and ready, but its preprocessing Python"
  echo "dependencies are not fully installed in the current environment."
  echo "Use a dedicated Python 3.11 environment and install the upstream"
  echo "preprocessing dependencies before running tools/prepare_teacher_cache.sh."
  exit 2
fi

echo
echo "teacher_source=$MEANVC2_DIR"
echo "teacher_commit=$ACTUAL_COMMIT"
echo "source bootstrap complete"
