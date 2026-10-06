from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from torch import Tensor, nn

from training.condition_fusion import ConditionFusion, ConditionFusionConfig
from training.contentnet import CausalContentNet, ContentNetConfig
from training.decoder import DecoderNet, DecoderNetConfig
from training.lite_vocoder import LiteVocoder, LiteVocoderConfig
from training.timbrenet import TimbreNet, TimbreNetConfig
from voxera import __version__
from voxera.deployment import (
    BundleManifest,
    ModelArtifact,
    PrecisionPolicy,
    TensorSpec,
    sha256_file,
)


class ContentStreamingWrapper(nn.Module):
    def __init__(self, model: CausalContentNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        return self.model.forward_chunk(features, cache)


class FusionExportWrapper(nn.Module):
    def __init__(self, model: ConditionFusion) -> None:
        super().__init__()
        self.model = model

    def forward(self, content: Tensor, pitch: Tensor, speaker: Tensor) -> Tensor:
        fused, _ = self.model(content, pitch, speaker)
        return fused


class DecoderStreamingWrapper(nn.Module):
    def __init__(self, model: DecoderNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, conditions: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        return self.model.forward_chunk(conditions, cache)


class VocoderStreamingWrapper(nn.Module):
    def __init__(self, model: LiteVocoder) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        mel: Tensor,
        cache: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.model.forward_chunk(mel, cache)


@dataclass(slots=True)
class CoreModels:
    contentnet: CausalContentNet
    timbrenet: TimbreNet
    condition_fusion: ConditionFusion
    decoder: DecoderNet
    lite_vocoder: LiteVocoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the Voxera M1 core as a verified ONNX deployment bundle"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path)
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--chunk-frames", type=int, default=8)
    parser.add_argument("--condition-frames", type=int, default=4)
    parser.add_argument("--reference-frames", type=int, default=32)
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.joint_checkpoint is None and not args.allow_untrained:
        raise ValueError(
            "a --joint-checkpoint is required unless --allow-untrained is "
            "explicitly enabled for architecture/CI validation"
        )
    if args.joint_checkpoint is not None and args.allow_untrained:
        raise ValueError("--joint-checkpoint and --allow-untrained are mutually exclusive")
    if min(
        args.chunk_frames,
        args.condition_frames,
        args.reference_frames,
        args.opset,
    ) <= 0:
        raise ValueError("frame counts and opset must be positive")

    models, weights_status, checkpoint_sha = _load_models(
        args.joint_checkpoint,
        allow_untrained=args.allow_untrained,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = (
        _export_contentnet(
            models.contentnet,
            output_dir / "contentnet.onnx",
            args.chunk_frames,
            args.opset,
        ),
        _export_timbrenet(
            models.timbrenet,
            output_dir / "timbrenet.onnx",
            args.reference_frames,
            args.opset,
        ),
        _export_fusion(
            models.condition_fusion,
            output_dir / "condition_fusion.onnx",
            args.condition_frames,
            args.opset,
        ),
        _export_decoder(
            models.decoder,
            output_dir / "decodernet.onnx",
            args.condition_frames,
            args.opset,
        ),
        _export_vocoder(
            models.lite_vocoder,
            output_dir / "lite_vocoder.onnx",
            args.chunk_frames,
            args.opset,
        ),
    )

    manifest = BundleManifest(
        voxera_version=__version__,
        weights_status=weights_status,
        sample_rate=16_000,
        acoustic_hop_ms=10,
        condition_hop_ms=40,
        source_checkpoint_sha256=checkpoint_sha,
        artifacts=artifacts,
    )
    manifest_path = output_dir / "bundle.json"
    manifest.write(manifest_path)

    print(f"bundle={output_dir}")
    print(f"manifest={manifest_path}")
    print(f"weights_status={weights_status}")
    for artifact in artifacts:
        print(f"{artifact.name}={artifact.file}:{artifact.sha256}")
    return 0


def _load_models(
    checkpoint: Path | None,
    *,
    allow_untrained: bool,
) -> tuple[CoreModels, str, str | None]:
    if checkpoint is None:
        if not allow_untrained:
            raise AssertionError("untrained model path must be explicit")
        return (
            CoreModels(
                contentnet=CausalContentNet(ContentNetConfig()).eval(),
                timbrenet=TimbreNet(TimbreNetConfig()).eval(),
                condition_fusion=ConditionFusion(ConditionFusionConfig()).eval(),
                decoder=DecoderNet(DecoderNetConfig()).eval(),
                lite_vocoder=LiteVocoder(LiteVocoderConfig()).eval(),
            ),
            "untrained-smoke",
            None,
        )

    checkpoint = checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("component") != "joint_refinement":
        raise ValueError(
            "deployment export requires a joint_refinement checkpoint, got "
            f"{payload.get('component')!r}"
        )

    configs = payload["model_config"]
    states = payload["models"]
    models = CoreModels(
        contentnet=CausalContentNet(
            ContentNetConfig(**configs["contentnet"])
        ).eval(),
        timbrenet=TimbreNet(
            TimbreNetConfig(**configs["timbrenet"])
        ).eval(),
        condition_fusion=ConditionFusion(
            ConditionFusionConfig(**configs["condition_fusion"])
        ).eval(),
        decoder=DecoderNet(
            DecoderNetConfig(**configs["decoder"])
        ).eval(),
        lite_vocoder=LiteVocoder(
            LiteVocoderConfig(**configs["lite_vocoder"])
        ).eval(),
    )
    models.contentnet.load_state_dict(states["contentnet"], strict=True)
    models.timbrenet.load_state_dict(states["timbrenet"], strict=True)
    models.condition_fusion.load_state_dict(states["condition_fusion"], strict=True)
    models.decoder.load_state_dict(states["decoder"], strict=True)
    models.lite_vocoder.load_state_dict(states["lite_vocoder"], strict=True)
    return models, "joint-checkpoint", sha256_file(checkpoint)


def _export_contentnet(
    model: CausalContentNet,
    path: Path,
    chunk_frames: int,
    opset: int,
) -> ModelArtifact:
    features = torch.zeros(1, chunk_frames, model.config.input_dim)
    cache = model.initial_cache(1)
    _export(
        ContentStreamingWrapper(model).eval(),
        (features, cache),
        path,
        input_names=("features", "cache"),
        output_names=("content", "next_cache"),
        dynamic_axes={
            "features": {0: "batch", 1: "frames"},
            "cache": {1: "batch"},
            "content": {0: "batch", 1: "frames"},
            "next_cache": {1: "batch"},
        },
        opset=opset,
    )
    layers, cache_frames, hidden = model.cache_shape
    return _artifact(
        "contentnet",
        path,
        opset,
        streaming=True,
        inputs=(
            TensorSpec("features", "float32", ("batch", "frames", model.config.input_dim)),
            TensorSpec("cache", "float32", (layers, "batch", cache_frames, hidden)),
        ),
        outputs=(
            TensorSpec("content", "float32", ("batch", "frames", model.config.output_dim)),
            TensorSpec("next_cache", "float32", (layers, "batch", cache_frames, hidden)),
        ),
    )


def _export_timbrenet(
    model: TimbreNet,
    path: Path,
    reference_frames: int,
    opset: int,
) -> ModelArtifact:
    features = torch.zeros(1, reference_frames, model.config.input_dim)
    _export(
        model,
        (features,),
        path,
        input_names=("features",),
        output_names=("speaker",),
        dynamic_axes={
            "features": {0: "batch", 1: "reference_frames"},
            "speaker": {0: "batch"},
        },
        opset=opset,
    )
    return _artifact(
        "timbrenet",
        path,
        opset,
        streaming=False,
        inputs=(
            TensorSpec(
                "features",
                "float32",
                ("batch", "reference_frames", model.config.input_dim),
            ),
        ),
        outputs=(
            TensorSpec(
                "speaker",
                "float32",
                ("batch", model.config.embedding_dim),
            ),
        ),
    )


def _export_fusion(
    model: ConditionFusion,
    path: Path,
    condition_frames: int,
    opset: int,
) -> ModelArtifact:
    cfg = model.config
    content = torch.zeros(1, condition_frames, cfg.content_dim)
    pitch = torch.zeros(1, condition_frames, cfg.pitch_dim)
    speaker = torch.zeros(1, cfg.speaker_dim)
    _export(
        FusionExportWrapper(model).eval(),
        (content, pitch, speaker),
        path,
        input_names=("content", "pitch", "speaker"),
        output_names=("condition",),
        dynamic_axes={
            "content": {0: "batch", 1: "condition_frames"},
            "pitch": {0: "batch", 1: "condition_frames"},
            "speaker": {0: "batch"},
            "condition": {0: "batch", 1: "condition_frames"},
        },
        opset=opset,
    )
    return _artifact(
        "condition_fusion",
        path,
        opset,
        streaming=True,
        inputs=(
            TensorSpec("content", "float32", ("batch", "condition_frames", cfg.content_dim)),
            TensorSpec("pitch", "float32", ("batch", "condition_frames", cfg.pitch_dim)),
            TensorSpec("speaker", "float32", ("batch", cfg.speaker_dim)),
        ),
        outputs=(
            TensorSpec(
                "condition",
                "float32",
                ("batch", "condition_frames", cfg.output_dim),
            ),
        ),
    )


def _export_decoder(
    model: DecoderNet,
    path: Path,
    condition_frames: int,
    opset: int,
) -> ModelArtifact:
    cfg = model.config
    conditions = torch.zeros(1, condition_frames, cfg.input_dim)
    cache = model.initial_cache(1)
    _export(
        DecoderStreamingWrapper(model).eval(),
        (conditions, cache),
        path,
        input_names=("conditions", "cache"),
        output_names=("mel", "next_cache"),
        dynamic_axes={
            "conditions": {0: "batch", 1: "condition_frames"},
            "cache": {1: "batch"},
            "mel": {0: "batch", 1: "mel_frames"},
            "next_cache": {1: "batch"},
        },
        opset=opset,
    )
    layers, cache_frames, hidden = model.cache_shape
    return _artifact(
        "decodernet",
        path,
        opset,
        streaming=True,
        inputs=(
            TensorSpec(
                "conditions",
                "float32",
                ("batch", "condition_frames", cfg.input_dim),
            ),
            TensorSpec("cache", "float32", (layers, "batch", cache_frames, hidden)),
        ),
        outputs=(
            TensorSpec("mel", "float32", ("batch", "mel_frames", cfg.output_mels)),
            TensorSpec("next_cache", "float32", (layers, "batch", cache_frames, hidden)),
        ),
    )


def _export_vocoder(
    model: LiteVocoder,
    path: Path,
    chunk_frames: int,
    opset: int,
) -> ModelArtifact:
    cfg = model.config
    mel = torch.zeros(1, chunk_frames, cfg.input_mels)
    cache = model.initial_cache(1)
    _export(
        VocoderStreamingWrapper(model).eval(),
        (mel, cache),
        path,
        input_names=("mel", "cache"),
        output_names=("log_magnitude", "phase", "next_cache"),
        dynamic_axes={
            "mel": {0: "batch", 1: "mel_frames"},
            "cache": {1: "batch"},
            "log_magnitude": {0: "batch", 1: "mel_frames"},
            "phase": {0: "batch", 1: "mel_frames"},
            "next_cache": {1: "batch"},
        },
        opset=opset,
    )
    layers, cache_frames, hidden = model.cache_shape
    frequency_bins = cfg.frequency_bins
    return _artifact(
        "lite_vocoder",
        path,
        opset,
        streaming=True,
        inputs=(
            TensorSpec("mel", "float32", ("batch", "mel_frames", cfg.input_mels)),
            TensorSpec("cache", "float32", (layers, "batch", cache_frames, hidden)),
        ),
        outputs=(
            TensorSpec(
                "log_magnitude",
                "float32",
                ("batch", "mel_frames", frequency_bins),
            ),
            TensorSpec(
                "phase",
                "float32",
                ("batch", "mel_frames", frequency_bins),
            ),
            TensorSpec("next_cache", "float32", (layers, "batch", cache_frames, hidden)),
        ),
    )


def _artifact(
    name: str,
    path: Path,
    opset: int,
    *,
    streaming: bool,
    inputs: tuple[TensorSpec, ...],
    outputs: tuple[TensorSpec, ...],
) -> ModelArtifact:
    return ModelArtifact(
        name=name,
        file=path.name,
        sha256=sha256_file(path),
        opset=opset,
        required=True,
        streaming=streaming,
        inputs=inputs,
        outputs=outputs,
        precision=PrecisionPolicy(
            baseline="fp32",
            candidates=("fp16", "int8_static"),
            calibration_required=("int8_static",),
        ),
    )


def _export(
    model: nn.Module,
    inputs: tuple[Tensor, ...],
    path: Path,
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    model_proto = onnx.load(path)
    onnx.checker.check_model(model_proto)


if __name__ == "__main__":
    raise SystemExit(main())
