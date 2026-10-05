from __future__ import annotations

import argparse
from pathlib import Path

from voxera.teacher_cache import build_teacher_cache_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate MeanVC2 teacher outputs and build Voxera manifest"
    )
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_teacher_cache_manifest(args.wav_dir, args.cache_dir)
    print(f"utterances={summary.utterances}")
    print(f"bn_frames={summary.bn_frames}")
    print(f"mel_frames={summary.mel_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
