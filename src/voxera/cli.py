from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .audio import load_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxera")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a 16-bit PCM WAV")
    inspect_parser.add_argument("input", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        audio = load_wav(args.input)
        peak = float(np.max(np.abs(audio.samples), initial=0.0))
        print(f"sample_rate={audio.sample_rate}")
        print(f"samples={audio.samples.size}")
        print(f"duration_seconds={audio.duration_seconds:.6f}")
        print(f"peak={peak:.6f}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
