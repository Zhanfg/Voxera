from __future__ import annotations

import argparse
from pathlib import Path

from voxera.teacher_cache import invalidate_stale_teacher_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Invalidate stale MeanVC2 teacher arrays before extraction"
    )
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = invalidate_stale_teacher_cache(args.wav_dir, args.cache_dir)
    print(f"stale_utterances={len(summary.stale_utterances)}")
    print(f"invalidated_outputs={summary.invalidated_outputs}")
    print(f"teacher_reset={str(summary.teacher_reset).lower()}")
    if summary.stale_utterances:
        print("stale_ids=" + ",".join(summary.stale_utterances))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
