from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from .audio import load_wav
from .backends.audio_cpp import (
    AudioCppMeanVC2Backend,
    default_reference_paths,
    format_doctor_report,
)


def _reference_defaults() -> tuple[Path, Path]:
    repo_root = Path(os.environ.get("VOXERA_REPO_ROOT", Path.cwd()))
    default_cli, default_model = default_reference_paths(repo_root)
    cli = Path(os.environ.get("VOXERA_AUDIOCPP_CLI", default_cli))
    model = Path(os.environ.get("VOXERA_MEANVC2_MODEL", default_model))
    return cli, model


def _add_reference_paths(parser: argparse.ArgumentParser) -> None:
    default_cli, default_model = _reference_defaults()
    parser.add_argument(
        "--audiocpp-cli",
        type=Path,
        default=default_cli,
        help="path to the audio.cpp CLI binary",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help="path to the MeanVC2 GGUF model",
    )
    parser.add_argument(
        "--compute-backend",
        choices=["best", "cpu", "cuda", "vulkan", "metal"],
        default="best",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxera")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a 16-bit PCM WAV")
    inspect_parser.add_argument("input", type=Path)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="run or inspect the external M1 reference baseline",
    )
    baseline_subparsers = baseline_parser.add_subparsers(
        dest="baseline_command",
        required=True,
    )

    doctor_parser = baseline_subparsers.add_parser(
        "doctor",
        help="verify the MeanVC2 reference runtime and model",
    )
    _add_reference_paths(doctor_parser)

    convert_parser = baseline_subparsers.add_parser(
        "convert",
        help="convert source speech into a target reference voice",
    )
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("target", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--timeout", type=float, default=None)
    _add_reference_paths(convert_parser)

    return parser


def _reference_backend(args: argparse.Namespace) -> AudioCppMeanVC2Backend:
    return AudioCppMeanVC2Backend(
        args.audiocpp_cli,
        args.model,
        compute_backend=args.compute_backend,
    )


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

    if args.command == "baseline" and args.baseline_command == "doctor":
        report = _reference_backend(args).doctor()
        for line in format_doctor_report(report):
            print(line)
        return 0 if report.ready else 1

    if args.command == "baseline" and args.baseline_command == "convert":
        result = _reference_backend(args).convert(
            args.source,
            args.target,
            args.output,
            timeout_seconds=args.timeout,
        )
        print(f"output={result.output_path}")
        print(f"elapsed_seconds={result.elapsed_seconds:.6f}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
