#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ASR_ROOT="$ROOT_DIR/.cache/asr"
SHERPA_DIR="$ASR_ROOT/sherpa-onnx"
WHISPER_DIR="$ASR_ROOT/whisper.cpp"

SHERPA_REPO="https://github.com/k2-fsa/sherpa-onnx.git"
SHERPA_TAG="v1.13.8"
WHISPER_REPO="https://github.com/ggml-org/whisper.cpp.git"
WHISPER_TAG="v1.9.4"

BACKENDS="${VOXERA_ASR_BACKENDS:-sherpa}"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

checkout_tag() {
  directory="$1"
  repository="$2"
  tag="$3"

  if [ ! -d "$directory/.git" ]; then
    git init "$directory"
    git -C "$directory" remote add origin "$repository"
  fi

  git -C "$directory" fetch --depth 1 origin "refs/tags/$tag"
  git -C "$directory" checkout --detach FETCH_HEAD

  actual=$(git -C "$directory" describe --tags --exact-match HEAD 2>/dev/null || true)
  if [ "$actual" != "$tag" ]; then
    echo "failed to pin $directory to $tag" >&2
    exit 1
  fi
}

require git
mkdir -p "$ASR_ROOT"

case ",$BACKENDS," in
  *,sherpa,*|*,both,*)
    echo "[ASR] pinning sherpa-onnx $SHERPA_TAG"
    checkout_tag "$SHERPA_DIR" "$SHERPA_REPO" "$SHERPA_TAG"
    ;;
esac

case ",$BACKENDS," in
  *,whisper,*|*,both,*)
    echo "[ASR] pinning whisper.cpp $WHISPER_TAG"
    checkout_tag "$WHISPER_DIR" "$WHISPER_REPO" "$WHISPER_TAG"
    ;;
esac

case ",$BACKENDS," in
  *,sherpa,*|*,whisper,*|*,both,*)
    ;;
  *)
    echo "unsupported VOXERA_ASR_BACKENDS=$BACKENDS" >&2
    echo "supported: sherpa, whisper, both" >&2
    exit 1
    ;;
esac

cat > "$ASR_ROOT/provenance.txt" <<EOF
sherpa_onnx_tag=$SHERPA_TAG
sherpa_onnx_repo=$SHERPA_REPO
whisper_cpp_tag=$WHISPER_TAG
whisper_cpp_repo=$WHISPER_REPO
selected=$BACKENDS
EOF

echo
echo "ASR sources ready under $ASR_ROOT"
echo "selected=$BACKENDS"
echo "No ASR model weights were downloaded."
