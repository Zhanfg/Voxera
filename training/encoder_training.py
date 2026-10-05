from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from training.contentnet import CausalContentNet
from training.contentnet_loss import ContentDistillationLossConfig, content_distillation_loss
from training.timbrenet import TimbreNet
from training.timbrenet_loss import TimbreDistillationLossConfig, timbre_distillation_loss
from voxera.audio import load_wav, resample_linear
from voxera.features import LogMelFrontend
from voxera.teacher_cache import (
    MEANVC2_COMMIT,
    TeacherCacheRecord,
    load_teacher_cache_manifest,
    resolve_teacher_cache_path,
)


@dataclass(frozen=True, slots=True)
class EncoderTrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_accumulation: int = 4
    max_grad_norm: float = 5.0
    timbre_crop_frames: int = 400
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
        if self.timbre_crop_frames <= 0:
            raise ValueError("timbre_crop_frames must be positive")


@dataclass(frozen=True, slots=True)
class TrainSummary:
    component: str
    epoch: int
    records: int
    optimizer_steps: int
    mean_loss: float
    checkpoint: Path


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def train_contentnet(
    manifest_path: Path,
    output_dir: Path,
    *,
    epochs: int,
    device: torch.device,
    config: EncoderTrainConfig | None = None,
    max_records: int | None = None,
) -> list[TrainSummary]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    cfg = config or EncoderTrainConfig()
    records = load_teacher_cache_manifest(manifest_path)
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise ValueError("no teacher records selected")

    model = CausalContentNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loss_config = ContentDistillationLossConfig()
    frontend = LogMelFrontend()
    rng = random.Random(cfg.seed)
    global_step = 0
    summaries: list[TrainSummary] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(records)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        seen = 0
        optimizer_steps = 0

        for index, record in enumerate(order, start=1):
            features = _native_features(record, manifest_path, frontend, device)
            teacher = _teacher_bn(record, manifest_path, device)

            dense = model(features.unsqueeze(0))
            available_student = dense[:, loss_config.offset :: loss_config.stride].shape[1]
            common = min(available_student, teacher.shape[0])
            if common <= 0:
                continue

            dense_frames = (
                loss_config.offset
                + (common - 1) * loss_config.stride
                + 1
            )
            aligned_dense = dense[:, :dense_frames]
            aligned_teacher = teacher[:common].unsqueeze(0)

            loss, _ = content_distillation_loss(
                aligned_dense,
                aligned_teacher,
                loss_config,
            )
            loss.backward()

            total_loss += float(loss.detach())
            seen += 1
            global_step += 1

            should_step = (
                index % cfg.gradient_accumulation == 0
                or index == len(order)
            )
            if should_step:
                accumulated = (
                    cfg.gradient_accumulation
                    if index % cfg.gradient_accumulation == 0
                    else index % cfg.gradient_accumulation
                )
                _average_gradients(model.parameters(), accumulated)
                clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        if seen == 0:
            raise RuntimeError("no ContentNet records produced aligned teacher frames")

        checkpoint = output_dir / f"contentnet-epoch{epoch:03d}.pt"
        _save_checkpoint(
            checkpoint,
            component="contentnet",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            model_config=asdict(model.config),
            train_config=asdict(cfg),
            manifest_path=manifest_path,
        )
        summaries.append(
            TrainSummary(
                component="contentnet",
                epoch=epoch,
                records=seen,
                optimizer_steps=optimizer_steps,
                mean_loss=total_loss / seen,
                checkpoint=checkpoint,
            )
        )

    return summaries


def train_timbrenet(
    manifest_path: Path,
    output_dir: Path,
    *,
    epochs: int,
    device: torch.device,
    config: EncoderTrainConfig | None = None,
    max_records: int | None = None,
) -> list[TrainSummary]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    cfg = config or EncoderTrainConfig()
    records = load_teacher_cache_manifest(manifest_path)
    if max_records is not None:
        records = records[:max_records]
    if not records:
        raise ValueError("no teacher records selected")

    model = TimbreNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loss_config = TimbreDistillationLossConfig()
    frontend = LogMelFrontend()
    rng = random.Random(cfg.seed + 17)
    global_step = 0
    summaries: list[TrainSummary] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(records)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        seen = 0
        optimizer_steps = 0

        for index, record in enumerate(order, start=1):
            features = _native_features(record, manifest_path, frontend, device)
            teacher = _teacher_speaker(record, manifest_path, device).unsqueeze(0)

            first = _random_crop(features, cfg.timbre_crop_frames, rng)
            second = _random_crop(features, cfg.timbre_crop_frames, rng)
            student = model(first.unsqueeze(0))
            second_view = model(second.unsqueeze(0))

            loss, _ = timbre_distillation_loss(
                student,
                teacher,
                second_view=second_view,
                config=loss_config,
            )
            loss.backward()

            total_loss += float(loss.detach())
            seen += 1
            global_step += 1

            should_step = (
                index % cfg.gradient_accumulation == 0
                or index == len(order)
            )
            if should_step:
                accumulated = (
                    cfg.gradient_accumulation
                    if index % cfg.gradient_accumulation == 0
                    else index % cfg.gradient_accumulation
                )
                _average_gradients(model.parameters(), accumulated)
                clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        checkpoint = output_dir / f"timbrenet-epoch{epoch:03d}.pt"
        _save_checkpoint(
            checkpoint,
            component="timbrenet",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            model_config=asdict(model.config),
            train_config=asdict(cfg),
            manifest_path=manifest_path,
        )
        summaries.append(
            TrainSummary(
                component="timbrenet",
                epoch=epoch,
                records=seen,
                optimizer_steps=optimizer_steps,
                mean_loss=total_loss / seen,
                checkpoint=checkpoint,
            )
        )

    return summaries


def _average_gradients(
    parameters,
    accumulated: int,
) -> None:
    if accumulated <= 0:
        raise ValueError("accumulated must be positive")
    scale = 1.0 / accumulated
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def _native_features(
    record: TeacherCacheRecord,
    manifest_path: Path,
    frontend: LogMelFrontend,
    device: torch.device,
) -> Tensor:
    wav_path = resolve_teacher_cache_path(manifest_path, record.wav)
    audio = load_wav(wav_path)
    audio = resample_linear(audio, frontend.config.sample_rate)
    features = frontend.extract(audio.samples).values
    if features.shape[0] == 0:
        raise ValueError(f"{record.utterance_id}: source is too short for fbank")
    return torch.from_numpy(features).to(device=device, dtype=torch.float32)


def _teacher_bn(
    record: TeacherCacheRecord,
    manifest_path: Path,
    device: torch.device,
) -> Tensor:
    path = resolve_teacher_cache_path(manifest_path, record.bn)
    array = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    return torch.from_numpy(array).to(device=device)


def _teacher_speaker(
    record: TeacherCacheRecord,
    manifest_path: Path,
    device: torch.device,
) -> Tensor:
    path = resolve_teacher_cache_path(manifest_path, record.speaker)
    array = np.load(path, allow_pickle=False).astype(np.float32, copy=False).reshape(-1)
    tensor = torch.from_numpy(array).to(device=device)
    return torch.nn.functional.normalize(tensor, dim=-1, eps=1e-8)


def _random_crop(
    features: Tensor,
    max_frames: int,
    rng: random.Random,
) -> Tensor:
    if features.shape[0] <= max_frames:
        return features
    start = rng.randrange(0, features.shape[0] - max_frames + 1)
    return features[start : start + max_frames]


def _save_checkpoint(
    path: Path,
    *,
    component: str,
    model: torch.nn.Module,
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
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    }


def summary_json(summary: TrainSummary) -> str:
    payload = {
        **asdict(summary),
        "checkpoint": str(summary.checkpoint),
    }
    return json.dumps(payload, ensure_ascii=False)
