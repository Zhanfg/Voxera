#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REF_DIR="$ROOT_DIR/.cache/reference"
AUDIOCPP_DIR="$REF_DIR/audio.cpp"
BUILD_DIR="$AUDIOCPP_DIR/build"
MODELS_DIR="$REF_DIR/models"
PACKAGE_ID="meanvc2_120ms_40ms_f32"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require git
require cmake

mkdir -p "$REF_DIR" "$MODELS_DIR"

if [ ! -d "$AUDIOCPP_DIR/.git" ]; then
  echo "[1/4] cloning audio.cpp reference runtime"
  git clone --depth 1 --recurse-submodules --shallow-submodules     https://github.com/0xShug0/audio.cpp.git "$AUDIOCPP_DIR"
else
  echo "[1/4] using existing audio.cpp checkout"
fi

git -C "$AUDIOCPP_DIR" rev-parse HEAD > "$REF_DIR/audio.cpp.commit"

echo "[2/4] configuring minimal MeanVC2 build"
cmake -S "$AUDIOCPP_DIR" -B "$BUILD_DIR"   -DCMAKE_BUILD_TYPE=Release   -DAUDIOCPP_MODEL_SET=custom   -DAUDIOCPP_MODELS=meanvc2   -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON

if command -v nproc >/dev/null 2>&1; then
  JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
  JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 2)
else
  JOBS=2
fi

echo "[3/4] building reference CLI and model manager"
cmake --build "$BUILD_DIR" --config Release   --target audiocpp_cli audiocpp_model_manager --parallel "$JOBS"

CLI="$BUILD_DIR/bin/audiocpp_cli"
MANAGER="$BUILD_DIR/bin/audiocpp_model_manager"
if [ ! -x "$CLI" ] || [ ! -x "$MANAGER" ]; then
  echo "reference binaries were not produced at the expected paths" >&2
  exit 1
fi

echo "[4/4] installing MeanVC2 model package"
"$MANAGER" install "$PACKAGE_ID" --models-dir "$MODELS_DIR"

MODEL=$(find "$MODELS_DIR" -type f -name 'meanvc2-120ms-40ms-fp32.gguf' -print -quit)
if [ -z "$MODEL" ]; then
  echo "MeanVC2 package installed, but expected GGUF was not found" >&2
  exit 1
fi

cat > "$REF_DIR/paths.env" <<EOF
VOXERA_AUDIOCPP_CLI=$CLI
VOXERA_MEANVC2_MODEL=$MODEL
EOF

echo
echo "reference backend ready"
echo "audio.cpp commit: $(cat "$REF_DIR/audio.cpp.commit")"
echo "CLI: $CLI"
echo "model: $MODEL"
echo
echo "Run:"
echo "  voxera baseline doctor"
echo "  voxera baseline convert source.wav target.wav converted.wav"
