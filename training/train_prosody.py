from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from training.prosody_training import ProsodyTrainConfig, train_prosodynet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap Voxera ProsodyNet from native acoustic descriptors"
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("data/train_wavs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-files", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    config = ProsodyTrainConfig(
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
    )
    print(f"device={device}")
    print(f"wav_dir={args.wav_dir.resolve()}")

    for summary in train_prosodynet(
        args.wav_dir,
        args.output_dir,
        epochs=args.epochs,
        device=device,
        config=config,
        max_files=args.max_files,
    ):
        payload = asdict(summary)
        payload["checkpoint"] = str(summary.checkpoint)
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
