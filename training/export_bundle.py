from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import Tensor, nn

from training.condition_fusion import ConditionFusion
from training.contentnet import CausalContentNet
from training.decoder import DecoderNet
from training.lite_vocoder import LiteVocoder
from training.prosody_conditioner import ProsodyConditioner
from training.prosodynet import ProsodyNet
from training.semantic_conditioner import SemanticConditioner
from training.timbrenet import TimbreNet


ONNX_OPSET = 17
BUNDLE_SCHEMA = 1


class ContentExport(nn.Module):
    def __init__(self, model: CausalContentNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        return self.model.forward_chunk(features, cache)


class TimbreExport(nn.Module):
    def __init__(self, model: TimbreNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: Tensor) -> Tensor:
        return self.model(features)


class FusionExport(nn.Module):
    def __init__(self, model: ConditionFusion) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        content: Tensor,
        pitch: Tensor,
        speaker: Tensor,
    ) -> Tensor:
        fused, _ = self.model(content, pitch, speaker)
        return fused


class DecoderExport(nn.Module):
    def __init__(self, model: DecoderNet) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        conditions: Tensor,
        cache: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.model.forward_chunk(conditions, cache)


class VocoderExport(nn.Module):
    def __init__(self, model: LiteVocoder) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        mel: Tensor,
        cache: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.model.forward_chunk(mel, cache)


class ProsodyExport(nn.Module):
    def __init__(self, model: ProsodyNet) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        acoustic: Tensor,
        pitch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.model(acoustic, pitch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a complete Voxera edge ONNX bundle"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/edge_bundle"),
    )
    parser.add_argument("--chunk-frames", type=int, default=16)
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Allow random/default weights for export-contract testing only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")

    manifest = export_edge_bundle(
        args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        chunk_frames=args.chunk_frames,
        allow_untrained=args.allow_untrained,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def export_edge_bundle(
    output_dir: Path,
    *,
    checkpoint_dir: Path,
    chunk_frames: int = 16,
    allow_untrained: bool = False,
) -> dict[str, Any]:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    output_dir = output_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, nn.Module] = {
        "contentnet": CausalContentNet().eval(),
        "timbrenet": TimbreNet().eval(),
        "condition_fusion": ConditionFusion().eval(),
        "decoder": DecoderNet().eval(),
        "lite_vocoder": LiteVocoder().eval(),
        "prosodynet": ProsodyNet().eval(),
        "prosody_conditioner": ProsodyConditioner().eval(),
        "semantic_conditioner": SemanticConditioner().eval(),
    }
    sources = _load_available_checkpoints(models, checkpoint_dir)
    trained = {name: source is not None for name, source in sources.items()}
    required_trained = (
        "contentnet",
        "timbrenet",
        "condition_fusion",
        "decoder",
        "lite_vocoder",
    )
    missing_required = [name for name in required_trained if not trained[name]]
    if missing_required and not allow_untrained:
        joined = ", ".join(missing_required)
        raise RuntimeError(
            "refusing to export a deployable bundle with untrained core models: "
            f"{joined}. Train M1 first or use --allow-untrained for export-contract "
            "testing only."
        )

    exports: dict[str, dict[str, Any]] = {}

    contentnet = models["contentnet"]
    assert isinstance(contentnet, CausalContentNet)
    exports["contentnet"] = _export_model(
        output_dir / "contentnet.onnx",
        ContentExport(contentnet),
        (
            torch.zeros(1, chunk_frames, contentnet.config.input_dim),
            contentnet.initial_cache(1),
        ),
        input_names=["features", "cache"],
        output_names=["content", "next_cache"],
        dynamic_axes={
            "features": {0: "batch", 1: "frames"},
            "cache": {1: "batch"},
            "content": {0: "batch", 1: "frames"},
            "next_cache": {1: "batch"},
        },
        source=sources["contentnet"],
        trained=trained["contentnet"],
        parameters=_parameters(contentnet),
        contract={
            "cadence_ms": 10,
            "input_dim": contentnet.config.input_dim,
            "output_dim": contentnet.config.output_dim,
            "cache_shape": list(contentnet.cache_shape),
        },
    )

    timbrenet = models["timbrenet"]
    assert isinstance(timbrenet, TimbreNet)
    exports["timbrenet"] = _export_model(
        output_dir / "timbrenet.onnx",
        TimbreExport(timbrenet),
        (torch.zeros(1, max(chunk_frames, 32), timbrenet.config.input_dim),),
        input_names=["features"],
        output_names=["speaker"],
        dynamic_axes={
            "features": {0: "batch", 1: "frames"},
            "speaker": {0: "batch"},
        },
        source=sources["timbrenet"],
        trained=trained["timbrenet"],
        parameters=_parameters(timbrenet),
        contract={
            "input_dim": timbrenet.config.input_dim,
            "embedding_dim": timbrenet.config.embedding_dim,
        },
    )

    fusion = models["condition_fusion"]
    assert isinstance(fusion, ConditionFusion)
    exports["condition_fusion"] = _export_model(
        output_dir / "condition_fusion.onnx",
        FusionExport(fusion),
        (
            torch.zeros(1, chunk_frames, fusion.config.content_dim),
            torch.zeros(1, chunk_frames, fusion.config.pitch_dim),
            torch.zeros(1, fusion.config.speaker_dim),
        ),
        input_names=["content", "pitch", "speaker"],
        output_names=["conditions"],
        dynamic_axes={
            "content": {0: "batch", 1: "frames"},
            "pitch": {0: "batch", 1: "frames"},
            "speaker": {0: "batch"},
            "conditions": {0: "batch", 1: "frames"},
        },
        source=sources["condition_fusion"],
        trained=trained["condition_fusion"],
        parameters=_parameters(fusion),
        contract={
            "cadence_ms": 40,
            "content_dim": fusion.config.content_dim,
            "pitch_dim": fusion.config.pitch_dim,
            "speaker_dim": fusion.config.speaker_dim,
            "output_dim": fusion.config.output_dim,
        },
    )

    prosodynet = models["prosodynet"]
    assert isinstance(prosodynet, ProsodyNet)
    exports["prosodynet"] = _export_model(
        output_dir / "prosodynet.onnx",
        ProsodyExport(prosodynet),
        (
            torch.zeros(1, max(chunk_frames, 32), prosodynet.config.acoustic_dim),
            torch.zeros(1, max(chunk_frames, 32), prosodynet.config.pitch_dim),
        ),
        input_names=["acoustic", "pitch"],
        output_names=[
            "local_prosody",
            "global_style",
            "local_descriptor",
            "global_descriptor",
        ],
        dynamic_axes={
            "acoustic": {0: "batch", 1: "frames"},
            "pitch": {0: "batch", 1: "frames"},
            "local_prosody": {0: "batch", 1: "frames"},
            "global_style": {0: "batch"},
            "local_descriptor": {0: "batch", 1: "frames"},
            "global_descriptor": {0: "batch"},
        },
        source=sources["prosodynet"],
        trained=trained["prosodynet"],
        parameters=_parameters(prosodynet),
        contract={
            "cadence_ms": 10,
            "acoustic_dim": prosodynet.config.acoustic_dim,
            "pitch_dim": prosodynet.config.pitch_dim,
            "local_dim": prosodynet.config.local_dim,
            "global_dim": prosodynet.config.global_dim,
            "offline_global_pooling": True,
        },
    )

    prosody_conditioner = models["prosody_conditioner"]
    assert isinstance(prosody_conditioner, ProsodyConditioner)
    exports["prosody_conditioner"] = _export_model(
        output_dir / "prosody_conditioner.onnx",
        prosody_conditioner,
        (
            torch.zeros(1, chunk_frames, prosody_conditioner.config.base_dim),
            torch.zeros(1, chunk_frames, prosody_conditioner.config.local_dim),
            torch.zeros(1, prosody_conditioner.config.global_dim),
        ),
        input_names=["base", "local_prosody", "global_style"],
        output_names=["conditions"],
        dynamic_axes={
            "base": {0: "batch", 1: "frames"},
            "local_prosody": {0: "batch", 1: "frames"},
            "global_style": {0: "batch"},
            "conditions": {0: "batch", 1: "frames"},
        },
        source=sources["prosody_conditioner"],
        trained=trained["prosody_conditioner"],
        parameters=_parameters(prosody_conditioner),
        contract=asdict(prosody_conditioner.config),
    )

    semantic_conditioner = models["semantic_conditioner"]
    assert isinstance(semantic_conditioner, SemanticConditioner)
    exports["semantic_conditioner"] = _export_model(
        output_dir / "semantic_conditioner.onnx",
        semantic_conditioner,
        (
            torch.zeros(1, chunk_frames, semantic_conditioner.config.base_dim),
            torch.zeros(1, semantic_conditioner.config.semantic_dim),
            torch.ones(1),
        ),
        input_names=["base", "semantic_features", "semantic_confidence"],
        output_names=["conditions"],
        dynamic_axes={
            "base": {0: "batch", 1: "frames"},
            "semantic_features": {0: "batch"},
            "semantic_confidence": {0: "batch"},
            "conditions": {0: "batch", 1: "frames"},
        },
        source=sources["semantic_conditioner"],
        trained=trained["semantic_conditioner"],
        parameters=_parameters(semantic_conditioner),
        contract=asdict(semantic_conditioner.config),
    )

    decoder = models["decoder"]
    assert isinstance(decoder, DecoderNet)
    exports["decoder"] = _export_model(
        output_dir / "decoder.onnx",
        DecoderExport(decoder),
        (
            torch.zeros(1, chunk_frames, decoder.config.input_dim),
            decoder.initial_cache(1),
        ),
        input_names=["conditions", "cache"],
        output_names=["mel", "next_cache"],
        dynamic_axes={
            "conditions": {0: "batch", 1: "condition_frames"},
            "cache": {1: "batch"},
            "mel": {0: "batch", 1: "mel_frames"},
            "next_cache": {1: "batch"},
        },
        source=sources["decoder"],
        trained=trained["decoder"],
        parameters=_parameters(decoder),
        contract={
            "input_cadence_ms": 40,
            "output_cadence_ms": 10,
            "input_dim": decoder.config.input_dim,
            "output_mels": decoder.config.output_mels,
            "upsample_factor": decoder.config.upsample_factor,
            "cache_shape": list(decoder.cache_shape),
        },
    )

    vocoder = models["lite_vocoder"]
    assert isinstance(vocoder, LiteVocoder)
    exports["lite_vocoder"] = _export_model(
        output_dir / "lite_vocoder.onnx",
        VocoderExport(vocoder),
        (
            torch.zeros(1, chunk_frames, vocoder.config.input_mels),
            vocoder.initial_cache(1),
        ),
        input_names=["mel", "cache"],
        output_names=["log_magnitude", "phase", "next_cache"],
        dynamic_axes={
            "mel": {0: "batch", 1: "frames"},
            "cache": {1: "batch"},
            "log_magnitude": {0: "batch", 1: "frames"},
            "phase": {0: "batch", 1: "frames"},
            "next_cache": {1: "batch"},
        },
        source=sources["lite_vocoder"],
        trained=trained["lite_vocoder"],
        parameters=_parameters(vocoder),
        contract={
            "cadence_ms": 10,
            "input_mels": vocoder.config.input_mels,
            "frequency_bins": vocoder.config.frequency_bins,
            "n_fft": vocoder.config.n_fft,
            "cache_shape": list(vocoder.cache_shape),
        },
    )

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "format": "voxera-edge-onnx",
        "onnx_opset": ONNX_OPSET,
        "precision": "fp32",
        "sample_rate": 16_000,
        "deployable": not missing_required,
        "untrained_export_allowed": allow_untrained,
        "missing_required_models": missing_required,
        "models": exports,
        "totals": {
            "unique_parameters": sum(
                _parameters(model)
                for model in models.values()
            ),
            "onnx_bytes": sum(item["bytes"] for item in exports.values()),
        },
        "notes": {
            "quantization": "not yet calibrated; real trained weights required",
            "prosodynet_global": "phrase-level/offline in M4 bundle; M5 adds streaming state",
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _export_model(
    path: Path,
    model: nn.Module,
    args: tuple[Tensor, ...],
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    source: dict[str, str] | None,
    trained: bool,
    parameters: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    torch.onnx.export(
        model,
        args,
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "parameters": parameters,
        "trained": trained,
        "weights_status": "checkpoint" if trained else "default_initialization",
        "source": source,
        "contract": contract,
    }


def _load_available_checkpoints(
    models: dict[str, nn.Module],
    checkpoint_dir: Path,
) -> dict[str, dict[str, str] | None]:
    sources: dict[str, dict[str, str] | None] = {
        name: None for name in models
    }
    if not checkpoint_dir.is_dir():
        return sources

    joint = _latest(checkpoint_dir, "joint-refinement-epoch*.pt")
    if joint is not None:
        payload = torch.load(joint, map_location="cpu", weights_only=False)
        states = payload.get("models", {})
        for name in (
            "contentnet",
            "timbrenet",
            "condition_fusion",
            "decoder",
            "lite_vocoder",
        ):
            state = states.get(name)
            if state is not None:
                models[name].load_state_dict(state, strict=True)
                sources[name] = _checkpoint_source(joint)

    staged = {
        "contentnet": ("contentnet-epoch*.pt", "model"),
        "timbrenet": ("timbrenet-epoch*.pt", "model"),
        "prosodynet": ("prosodynet-epoch*.pt", "model"),
    }
    for name, (pattern, key) in staged.items():
        if sources[name] is not None:
            continue
        path = _latest(checkpoint_dir, pattern)
        if path is None:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get(key)
        if state is not None:
            models[name].load_state_dict(state, strict=True)
            sources[name] = _checkpoint_source(path)

    acoustic = _latest(checkpoint_dir, "acoustic-generator-epoch*.pt")
    if acoustic is not None:
        payload = torch.load(acoustic, map_location="cpu", weights_only=False)
        states = payload.get("models", {})
        for name in ("condition_fusion", "decoder"):
            if sources[name] is None and name in states:
                models[name].load_state_dict(states[name], strict=True)
                sources[name] = _checkpoint_source(acoustic)

    vocoder = _latest(checkpoint_dir, "lite-vocoder-epoch*.pt")
    if vocoder is not None and sources["lite_vocoder"] is None:
        payload = torch.load(vocoder, map_location="cpu", weights_only=False)
        state = payload.get("models", {}).get("lite_vocoder")
        if state is not None:
            models["lite_vocoder"].load_state_dict(state, strict=True)
            sources["lite_vocoder"] = _checkpoint_source(vocoder)

    for name, pattern in (
        ("prosody_conditioner", "prosody-conditioner-epoch*.pt"),
        ("semantic_conditioner", "semantic-conditioner-epoch*.pt"),
    ):
        path = _latest(checkpoint_dir, pattern)
        if path is None:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model")
        if state is not None:
            models[name].load_state_dict(state, strict=True)
            sources[name] = _checkpoint_source(path)

    return sources


def _latest(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _checkpoint_source(path: Path) -> dict[str, str]:
    return {
        "file": path.name,
        "sha256": _sha256(path),
    }


def _parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
