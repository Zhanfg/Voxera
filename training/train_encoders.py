from __future__ import annotations

import argparse
from pathlib import Path

from training.encoder_training import (
    EncoderTrainConfig,
    resolve_device,
    summary_json,
    train_contentnet,
    train_timbrenet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Voxera compact encoders")
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
        choices=["content", "timbre", "both"],
        default="both",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-records", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    config = EncoderTrainConfig(
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
    )

    print(f"device={device}")
    print(f"manifest={args.manifest.resolve()}")

    if args.component in {"content", "both"}:
        summaries = train_contentnet(
            args.manifest,
            args.output_dir,
            epochs=args.epochs,
            device=device,
            config=config,
            max_records=args.max_records,
        )
        for summary in summaries:
            print(summary_json(summary))

    if args.component in {"timbre", "both"}:
        summaries = train_timbrenet(
            args.manifest,
            args.output_dir,
            epochs=args.epochs,
            device=device,
            config=config,
            max_records=args.max_records,
        )
        for summary in summaries:
            print(summary_json(summary))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
