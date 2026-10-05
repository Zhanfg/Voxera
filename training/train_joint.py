from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from training.joint_training import (
    JointTrainConfig,
    find_latest_staged_checkpoints,
    train_joint_refinement,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jointly refine the Voxera M1 graph")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/teacher_cache/manifest.jsonl"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--max-records", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    staged = find_latest_staged_checkpoints(args.checkpoint_dir)
    config = JointTrainConfig(
        learning_rate=args.learning_rate,
        gradient_accumulation=args.gradient_accumulation,
    )

    print(f"device={device}")
    print(f"manifest={args.manifest.resolve()}")
    print(f"contentnet={staged.contentnet}")
    print(f"timbrenet={staged.timbrenet}")
    print(f"acoustic_generator={staged.acoustic_generator}")
    print(f"lite_vocoder={staged.lite_vocoder}")

    summaries = train_joint_refinement(
        args.manifest,
        staged,
        args.output_dir,
        epochs=args.epochs,
        device=device,
        config=config,
        max_records=args.max_records,
    )
    for summary in summaries:
        payload = asdict(summary)
        payload["checkpoint"] = str(summary.checkpoint)
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
