from __future__ import annotations

import argparse
from pathlib import Path

import torch
from contentnet import CausalContentNet, ContentNetConfig
from torch import Tensor, nn


class StreamingExportWrapper(nn.Module):
    def __init__(self, model: CausalContentNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        return self.model.forward_chunk(features, cache)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Voxera ContentNet to ONNX")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=16)
    return parser


def load_model(checkpoint: Path | None) -> CausalContentNet:
    model = CausalContentNet(ContentNetConfig())
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
    return model.eval()


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")

    model = load_model(args.checkpoint)
    wrapper = StreamingExportWrapper(model).eval()
    features = torch.zeros(1, args.chunk_frames, model.config.input_dim)
    cache = model.initial_cache(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (features, cache),
        args.output,
        input_names=["features", "cache"],
        output_names=["content", "next_cache"],
        dynamic_axes={
            "features": {0: "batch", 1: "frames"},
            "cache": {1: "batch"},
            "content": {0: "batch", 1: "frames"},
            "next_cache": {1: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
