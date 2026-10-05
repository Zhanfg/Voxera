from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from training.condition_fusion import ConditionFusion
from training.decoder import DecoderNet
from training.decoder_loss import decoder_reconstruction_loss
from training.lite_vocoder import LiteVocoder
from training.vocoder_loss import vocoder_reconstruction_loss
from voxera.audio import load_wav, resample_linear
from voxera.conditioning import pitch_condition_features
from voxera.pitch import YinPitchExtractor
from voxera.teacher_cache import (
    MEANVC2_COMMIT,
    TeacherCacheRecord,
    load_teacher_cache_manifest,
    resolve_teacher_cache_path,
)


@dataclass(frozen=True, slots=True)
class GeneratorTrainConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    gradient_accumulation: int = 4
    max_grad_norm: float = 5.0
    decoder_max_condition_frames: int = 160
    vocoder_max_frames: int = 256
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.decoder_max_condition_frames <= 0:
            raise ValueError("decoder_max_condition_frames must be positive")
        if self.vocoder_max_frames <= 0:
            raise ValueError("vocoder_max_frames must be positive")


@dataclass(frozen=True, slots=True)
class GeneratorTrainSummary:
    component: str
    epoch: int
    records: int
    optimizer_steps: int
    mean_loss: float
    checkpoint: Path


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def train_acoustic_generator(
    manifest_path: Path,
    output_dir: Path,
    *,
    epochs: int,
    device: torch.device,
    config: GeneratorTrainConfig | None = None,
    max_records: int | None = None,
) -> list[GeneratorTrainSummary]:
    """Train ConditionFusion + DecoderNet from teacher BN/speaker/mel targets."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    cfg = config or GeneratorTrainConfig()
    records = _selected_records(manifest_path, max_records)
    fusion = ConditionFusion().to(device)
    decoder = DecoderNet().to(device)
    parameters = [*fusion.parameters(), *decoder.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    pitch_extractor = YinPitchExtractor()
    rng = random.Random(cfg.seed + 101)
    global_step = 0
    summaries: list[GeneratorTrainSummary] = []

    for epoch in range(1, epochs + 1):
        fusion.train()
        decoder.train()
        order = list(records)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        seen = 0
        optimizer_steps = 0
        accumulated = 0

        for record in order:
            sample = _acoustic_training_sample(
                record,
                manifest_path,
                device,
                pitch_extractor,
                cfg.decoder_max_condition_frames,
                rng,
            )
            if sample is None:
                continue
            content, pitch, speaker, target_mel = sample

            fused, _ = fusion(
                content.unsqueeze(0),
                pitch.unsqueeze(0),
                speaker.unsqueeze(0),
            )
            prediction = decoder(fused)
            loss, _ = decoder_reconstruction_loss(
                prediction,
                target_mel.unsqueeze(0),
            )
            (loss / cfg.gradient_accumulation).backward()

            total_loss += float(loss.detach())
            seen += 1
            accumulated += 1
            global_step += 1

            if accumulated == cfg.gradient_accumulation:
                clip_grad_norm_(parameters, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0

        if accumulated:
            clip_grad_norm_(parameters, cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if seen == 0:
            raise RuntimeError("no records produced an aligned acoustic training sample")

        checkpoint = output_dir / f"acoustic-generator-epoch{epoch:03d}.pt"
        _save_checkpoint(
            checkpoint,
            component="acoustic_generator",
            models={
                "condition_fusion": fusion,
                "decoder": decoder,
            },
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            model_config={
                "condition_fusion": asdict(fusion.config),
                "decoder": asdict(decoder.config),
            },
            train_config=asdict(cfg),
            manifest_path=manifest_path,
        )
        summaries.append(
            GeneratorTrainSummary(
                component="acoustic_generator",
                epoch=epoch,
                records=seen,
                optimizer_steps=optimizer_steps,
                mean_loss=total_loss / seen,
                checkpoint=checkpoint,
            )
        )

    return summaries


def train_lite_vocoder(
    manifest_path: Path,
    output_dir: Path,
    *,
    epochs: int,
    device: torch.device,
    config: GeneratorTrainConfig | None = None,
    max_records: int | None = None,
) -> list[GeneratorTrainSummary]:
    """Train LiteVocoder from teacher mel aligned to ground-truth source PCM."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    cfg = config or GeneratorTrainConfig()
    records = _selected_records(manifest_path, max_records)
    model = LiteVocoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    rng = random.Random(cfg.seed + 211)
    global_step = 0
    summaries: list[GeneratorTrainSummary] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(records)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        seen = 0
        optimizer_steps = 0
        accumulated = 0

        for record in order:
            sample = _vocoder_training_sample(
                record,
                manifest_path,
                device,
                cfg.vocoder_max_frames,
                rng,
            )
            if sample is None:
                continue
            mel, waveform = sample

            log_magnitude, phase = model(mel.unsqueeze(0))
            loss, _ = vocoder_reconstruction_loss(
                log_magnitude,
                phase,
                waveform.unsqueeze(0),
            )
            (loss / cfg.gradient_accumulation).backward()

            total_loss += float(loss.detach())
            seen += 1
            accumulated += 1
            global_step += 1

            if accumulated == cfg.gradient_accumulation:
                clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0

        if accumulated:
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if seen == 0:
            raise RuntimeError("no records produced an aligned vocoder training sample")

        checkpoint = output_dir / f"lite-vocoder-epoch{epoch:03d}.pt"
        _save_checkpoint(
            checkpoint,
            component="lite_vocoder",
            models={"lite_vocoder": model},
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            model_config={"lite_vocoder": asdict(model.config)},
            train_config=asdict(cfg),
            manifest_path=manifest_path,
        )
        summaries.append(
            GeneratorTrainSummary(
                component="lite_vocoder",
                epoch=epoch,
                records=seen,
                optimizer_steps=optimizer_steps,
                mean_loss=total_loss / seen,
                checkpoint=checkpoint,
            )
        )

    return summaries


def _selected_records(
    manifest_path: Path,
    max_records: int | None,
) -> list[TeacherCacheRecord]:
    records = load_teacher_cache_manifest(manifest_path)
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        records = records[:max_records]
    if not records:
        raise ValueError("no teacher records selected")
    return records


def _acoustic_training_sample(
    record: TeacherCacheRecord,
    manifest_path: Path,
    device: torch.device,
    pitch_extractor: YinPitchExtractor,
    max_condition_frames: int,
    rng: random.Random,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    content = _load_array(record.bn, manifest_path, device)
    mel = _load_array(record.mel, manifest_path, device)
    speaker = _load_array(record.speaker, manifest_path, device).reshape(-1)
    speaker = torch.nn.functional.normalize(speaker, dim=-1, eps=1e-8)

    audio = _load_resampled_audio(record, manifest_path)
    pitch_track = pitch_extractor.extract(audio)
    pitch = torch.from_numpy(pitch_condition_features(pitch_track)).to(device=device)
    pitch = pitch[3::4]

    condition_frames = min(
        content.shape[0],
        pitch.shape[0],
        mel.shape[0] // 4,
    )
    if condition_frames <= 0:
        return None

    crop = min(condition_frames, max_condition_frames)
    start = 0
    if condition_frames > crop:
        start = rng.randrange(0, condition_frames - crop + 1)
    end = start + crop

    mel_start = start * 4
    mel_end = end * 4
    return (
        content[start:end],
        pitch[start:end],
        speaker,
        mel[mel_start:mel_end],
    )


def _vocoder_training_sample(
    record: TeacherCacheRecord,
    manifest_path: Path,
    device: torch.device,
    max_frames: int,
    rng: random.Random,
) -> tuple[Tensor, Tensor] | None:
    mel = _load_array(record.mel, manifest_path, device)
    audio = _load_resampled_audio(record, manifest_path)

    available_frames = min(mel.shape[0], audio.size // 160)
    if available_frames <= 0:
        return None

    crop = min(available_frames, max_frames)
    start = 0
    if available_frames > crop:
        start = rng.randrange(0, available_frames - crop + 1)
    end = start + crop

    mel_crop = mel[start:end]
    sample_start = start * 160
    sample_end = end * 160
    waveform = torch.from_numpy(audio[sample_start:sample_end].copy()).to(
        device=device,
        dtype=torch.float32,
    )
    return mel_crop, waveform


def _load_array(
    stored_path: str,
    manifest_path: Path,
    device: torch.device,
) -> Tensor:
    path = resolve_teacher_cache_path(manifest_path, stored_path)
    array = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    return torch.from_numpy(array).to(device=device)


def _load_resampled_audio(
    record: TeacherCacheRecord,
    manifest_path: Path,
) -> np.ndarray:
    path = resolve_teacher_cache_path(manifest_path, record.wav)
    audio = resample_linear(load_wav(path), 16_000)
    return audio.samples


def _save_checkpoint(
    path: Path,
    *,
    component: str,
    models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    model_config: dict[str, object],
    train_config: dict[str, object],
    manifest_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "component": component,
        "epoch": epoch,
        "global_step": global_step,
        "teacher": {
            "name": "MeanVC2",
            "commit": MEANVC2_COMMIT,
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _file_sha256(manifest_path),
        },
        "model_config": model_config,
        "train_config": train_config,
        "models": {
            name: model.state_dict()
            for name, model in models.items()
        },
        "optimizer": optimizer.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_metadata(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "schema": payload["schema"],
        "component": payload["component"],
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "teacher": payload["teacher"],
        "manifest": payload["manifest"],
        "model_config": payload["model_config"],
        "train_config": payload["train_config"],
        "models": sorted(payload["models"]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
