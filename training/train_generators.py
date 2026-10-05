from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from training.generator_training import (
    GeneratorTrainConfig,
    resolve_device,
    train_acoustic_generator,
    train_lite_vocoder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Voxera M1 generator stages")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/teacher_cache/manifest.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
    )
    parser.add_argument(
        "--component",
        choices=["acoustic", "vocoder", "both"],
        default="both",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-records", type=int)
    return parser


def _print_summary(summary) -> None:
    payload = asdict(summary)
    payload["checkpoint"] = str(summary.checkpoint)
    print(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    config = GeneratorTrainConfig(
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
    )

    print(f"device={device}")
    print(f"manifest={args.manifest.resolve()}")

    if args.component in {"acoustic", "both"}:
        for summary in train_acoustic_generator(
            args.manifest,
            args.output_dir,
            epochs=args.epochs,
            device=device,
            config=config,
            max_records=args.max_records,
        ):
            _print_summary(summary)

    if args.component in {"vocoder", "both"}:
        for summary in train_lite_vocoder(
            args.manifest,
            args.output_dir,
            epochs=args.epochs,
            device=device,
            config=config,
            max_records=args.max_records,
        ):
            _print_summary(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
