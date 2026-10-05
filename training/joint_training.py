from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from training.condition_fusion import ConditionFusion, ConditionFusionConfig
from training.contentnet import CausalContentNet, ContentNetConfig
from training.contentnet_loss import content_distillation_loss
from training.decoder import DecoderNet, DecoderNetConfig
from training.decoder_loss import decoder_reconstruction_loss
from training.lite_vocoder import LiteVocoder, LiteVocoderConfig
from training.timbrenet import TimbreNet, TimbreNetConfig
from training.timbrenet_loss import timbre_distillation_loss
from training.vocoder_loss import vocoder_reconstruction_loss
from voxera.audio import load_wav, resample_linear
from voxera.conditioning import pitch_condition_features
from voxera.features import LogMelFrontend
from voxera.pitch import YinPitchExtractor
from voxera.teacher_cache import (
    MEANVC2_COMMIT,
    TeacherCacheRecord,
    load_teacher_cache_manifest,
    resolve_teacher_cache_path,
)


@dataclass(frozen=True, slots=True)
class StagedCheckpoints:
    contentnet: Path
    timbrenet: Path
    acoustic_generator: Path
    lite_vocoder: Path


@dataclass(frozen=True, slots=True)
class JointTrainConfig:
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    gradient_accumulation: int = 2
    max_grad_norm: float = 3.0
    max_condition_frames: int = 80
    content_weight: float = 0.25
    timbre_weight: float = 0.25
    mel_weight: float = 1.0
    waveform_weight: float = 1.0
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
        if self.max_condition_frames <= 0:
            raise ValueError("max_condition_frames must be positive")
        if min(
            self.content_weight,
            self.timbre_weight,
            self.mel_weight,
            self.waveform_weight,
        ) < 0.0:
            raise ValueError("loss weights must be non-negative")


@dataclass(frozen=True, slots=True)
class JointTrainSummary:
    epoch: int
    records: int
    optimizer_steps: int
    mean_loss: float
    checkpoint: Path


@dataclass(slots=True)
class JointModels:
    contentnet: CausalContentNet
    timbrenet: TimbreNet
    condition_fusion: ConditionFusion
    decoder: DecoderNet
    lite_vocoder: LiteVocoder

    def modules(self) -> tuple[torch.nn.Module, ...]:
        return (
            self.contentnet,
            self.timbrenet,
            self.condition_fusion,
            self.decoder,
            self.lite_vocoder,
        )

    def parameters(self):
        for module in self.modules():
            yield from module.parameters()


def find_latest_staged_checkpoints(directory: Path) -> StagedCheckpoints:
    directory = directory.resolve()
    return StagedCheckpoints(
        contentnet=_latest(directory, "contentnet-epoch*.pt"),
        timbrenet=_latest(directory, "timbrenet-epoch*.pt"),
        acoustic_generator=_latest(directory, "acoustic-generator-epoch*.pt"),
        lite_vocoder=_latest(directory, "lite-vocoder-epoch*.pt"),
    )


def load_staged_models(
    checkpoints: StagedCheckpoints,
    manifest_path: Path,
    device: torch.device,
) -> JointModels:
    manifest_sha = _file_sha256(manifest_path)

    content_payload = _load_payload(
        checkpoints.contentnet,
        expected_component="contentnet",
        manifest_sha=manifest_sha,
    )
    timbre_payload = _load_payload(
        checkpoints.timbrenet,
        expected_component="timbrenet",
        manifest_sha=manifest_sha,
    )
    acoustic_payload = _load_payload(
        checkpoints.acoustic_generator,
        expected_component="acoustic_generator",
        manifest_sha=manifest_sha,
    )
    vocoder_payload = _load_payload(
        checkpoints.lite_vocoder,
        expected_component="lite_vocoder",
        manifest_sha=manifest_sha,
    )

    contentnet = CausalContentNet(
        ContentNetConfig(**content_payload["model_config"])
    )
    timbrenet = TimbreNet(
        TimbreNetConfig(**timbre_payload["model_config"])
    )
    condition_fusion = ConditionFusion(
        ConditionFusionConfig(**acoustic_payload["model_config"]["condition_fusion"])
    )
    decoder = DecoderNet(
        DecoderNetConfig(**acoustic_payload["model_config"]["decoder"])
    )
    lite_vocoder = LiteVocoder(
        LiteVocoderConfig(**vocoder_payload["model_config"]["lite_vocoder"])
    )

    contentnet.load_state_dict(content_payload["model"], strict=True)
    timbrenet.load_state_dict(timbre_payload["model"], strict=True)
    condition_fusion.load_state_dict(
        acoustic_payload["models"]["condition_fusion"],
        strict=True,
    )
    decoder.load_state_dict(acoustic_payload["models"]["decoder"], strict=True)
    lite_vocoder.load_state_dict(
        vocoder_payload["models"]["lite_vocoder"],
        strict=True,
    )

    models = JointModels(
        contentnet=contentnet.to(device),
        timbrenet=timbrenet.to(device),
        condition_fusion=condition_fusion.to(device),
        decoder=decoder.to(device),
        lite_vocoder=lite_vocoder.to(device),
    )
    return models


def train_joint_refinement(
    manifest_path: Path,
    checkpoints: StagedCheckpoints,
    output_dir: Path,
    *,
    epochs: int,
    device: torch.device,
    config: JointTrainConfig | None = None,
    max_records: int | None = None,
) -> list[JointTrainSummary]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    cfg = config or JointTrainConfig()
    records = load_teacher_cache_manifest(manifest_path)
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        records = records[:max_records]
    if not records:
        raise ValueError("no teacher records selected")

    models = load_staged_models(checkpoints, manifest_path, device)
    parameters = list(models.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    frontend = LogMelFrontend()
    pitch_extractor = YinPitchExtractor()
    rng = random.Random(cfg.seed + 307)
    global_step = 0
    summaries: list[JointTrainSummary] = []

    for epoch in range(1, epochs + 1):
        for module in models.modules():
            module.train()
        order = list(records)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        seen = 0
        accumulated = 0
        optimizer_steps = 0

        for record in order:
            sample = _joint_sample(
                record,
                manifest_path,
                frontend,
                pitch_extractor,
                device,
                cfg.max_condition_frames,
                rng,
            )
            if sample is None:
                continue

            (
                dense_features,
                target_bn,
                pitch,
                target_speaker,
                target_mel,
                target_waveform,
            ) = sample

            dense_content = models.contentnet(dense_features.unsqueeze(0))
            available_40ms = dense_content[:, 3::4].shape[1]
            condition_frames = min(
                available_40ms,
                target_bn.shape[0],
                pitch.shape[0],
                target_mel.shape[0] // 4,
            )
            if condition_frames <= 0:
                continue

            condition_start = 0
            crop = min(condition_frames, cfg.max_condition_frames)
            if condition_frames > crop:
                condition_start = rng.randrange(0, condition_frames - crop + 1)
            condition_end = condition_start + crop

            dense_start = condition_start * 4
            dense_end = condition_end * 4 + 3
            dense_crop = dense_content[:, dense_start:dense_end]
            bn_crop = target_bn[condition_start:condition_end].unsqueeze(0)
            content_loss, _ = content_distillation_loss(dense_crop, bn_crop)

            reference = _reference_crop(
                dense_features,
                max_frames=max(crop * 4, 16),
                rng=rng,
            )
            speaker = models.timbrenet(reference.unsqueeze(0))
            timbre_loss, _ = timbre_distillation_loss(
                speaker,
                target_speaker.unsqueeze(0),
            )

            student_40ms = dense_content[:, 3::4]
            student_40ms = student_40ms[:, condition_start:condition_end]
            pitch_crop = pitch[condition_start:condition_end].unsqueeze(0)
            fused, _ = models.condition_fusion(
                student_40ms,
                pitch_crop,
                speaker,
            )
            predicted_mel = models.decoder(fused)

            mel_start = condition_start * 4
            mel_end = condition_end * 4
            mel_target = target_mel[mel_start:mel_end].unsqueeze(0)
            mel_loss, _ = decoder_reconstruction_loss(
                predicted_mel,
                mel_target,
            )

            log_magnitude, phase = models.lite_vocoder(predicted_mel)
            waveform_start = mel_start * 160
            waveform_end = mel_end * 160
            waveform_target = target_waveform[waveform_start:waveform_end]
            if waveform_target.shape[0] != predicted_mel.shape[1] * 160:
                continue
            waveform_loss, _ = vocoder_reconstruction_loss(
                log_magnitude,
                phase,
                waveform_target.unsqueeze(0),
            )

            loss = (
                cfg.content_weight * content_loss
                + cfg.timbre_weight * timbre_loss
                + cfg.mel_weight * mel_loss
                + cfg.waveform_weight * waveform_loss
            )
            loss.backward()

            total_loss += float(loss.detach())
            seen += 1
            accumulated += 1
            global_step += 1

            if accumulated == cfg.gradient_accumulation:
                _average_gradients(parameters, accumulated)
                clip_grad_norm_(parameters, cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0

        if accumulated:
            _average_gradients(parameters, accumulated)
            clip_grad_norm_(parameters, cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if seen == 0:
            raise RuntimeError("no records produced a valid joint-refinement sample")

        checkpoint = output_dir / f"joint-refinement-epoch{epoch:03d}.pt"
        _save_joint_checkpoint(
            checkpoint,
            models=models,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config=cfg,
            manifest_path=manifest_path,
            staged=checkpoints,
        )
        summaries.append(
            JointTrainSummary(
                epoch=epoch,
                records=seen,
                optimizer_steps=optimizer_steps,
                mean_loss=total_loss / seen,
                checkpoint=checkpoint,
            )
        )

    return summaries


def _joint_sample(
    record: TeacherCacheRecord,
    manifest_path: Path,
    frontend: LogMelFrontend,
    pitch_extractor: YinPitchExtractor,
    device: torch.device,
    max_condition_frames: int,
    rng: random.Random,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None:
    path = resolve_teacher_cache_path(manifest_path, record.wav)
    audio = resample_linear(load_wav(path), 16_000)
    features = frontend.extract(audio.samples).values
    if features.shape[0] == 0:
        return None

    pitch_track = pitch_extractor.extract(audio.samples)
    pitch = pitch_condition_features(pitch_track)[3::4]
    teacher_bn = _load_array(record.bn, manifest_path)
    teacher_mel = _load_array(record.mel, manifest_path)
    teacher_speaker = _load_array(record.speaker, manifest_path).reshape(-1)

    available_conditions = min(
        (features.shape[0] - 3 + 3) // 4,
        pitch.shape[0],
        teacher_bn.shape[0],
        teacher_mel.shape[0] // 4,
        audio.samples.size // (4 * 160),
    )
    if available_conditions <= 0:
        return None

    needed_dense = 3 + available_conditions * 4
    features = features[:needed_dense]
    pitch = pitch[:available_conditions]
    teacher_bn = teacher_bn[:available_conditions]
    teacher_mel = teacher_mel[: available_conditions * 4]

    max_waveform_samples = available_conditions * 4 * 160
    waveform = audio.samples[:max_waveform_samples]

    del max_condition_frames, rng
    return (
        torch.from_numpy(features.copy()).to(device=device, dtype=torch.float32),
        torch.from_numpy(teacher_bn.copy()).to(device=device, dtype=torch.float32),
        torch.from_numpy(pitch.copy()).to(device=device, dtype=torch.float32),
        torch.nn.functional.normalize(
            torch.from_numpy(teacher_speaker.copy()).to(
                device=device,
                dtype=torch.float32,
            ),
            dim=-1,
            eps=1e-8,
        ),
        torch.from_numpy(teacher_mel.copy()).to(device=device, dtype=torch.float32),
        torch.from_numpy(waveform.copy()).to(device=device, dtype=torch.float32),
    )


def _reference_crop(
    features: Tensor,
    *,
    max_frames: int,
    rng: random.Random,
) -> Tensor:
    if features.shape[0] <= max_frames:
        return features
    start = rng.randrange(0, features.shape[0] - max_frames + 1)
    return features[start : start + max_frames]


def _load_array(stored_path: str, manifest_path: Path) -> np.ndarray:
    path = resolve_teacher_cache_path(manifest_path, stored_path)
    return np.load(path, allow_pickle=False).astype(np.float32, copy=False)


def _load_payload(
    path: Path,
    *,
    expected_component: str,
    manifest_sha: str,
) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != 1:
        raise ValueError(f"{path}: unsupported checkpoint schema")
    if payload.get("component") != expected_component:
        raise ValueError(
            f"{path}: expected component {expected_component}, "
            f"got {payload.get('component')}"
        )
    teacher = payload.get("teacher", {})
    if teacher.get("commit") != MEANVC2_COMMIT:
        raise ValueError(f"{path}: teacher commit mismatch")
    manifest = payload.get("manifest", {})
    if manifest.get("sha256") != manifest_sha:
        raise ValueError(f"{path}: teacher manifest SHA-256 mismatch")
    return payload


def _latest(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no checkpoint matches {pattern} in {directory}")
    return matches[-1]


def _average_gradients(parameters, accumulated: int) -> None:
    scale = 1.0 / accumulated
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def _save_joint_checkpoint(
    path: Path,
    *,
    models: JointModels,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: JointTrainConfig,
    manifest_path: Path,
    staged: StagedCheckpoints,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged_paths = asdict(staged)
    payload = {
        "schema": 1,
        "component": "joint_refinement",
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
        "train_config": asdict(config),
        "staged": {
            name: {
                "path": str(checkpoint.resolve()),
                "sha256": _file_sha256(checkpoint),
            }
            for name, checkpoint in staged_paths.items()
        },
        "model_config": {
            "contentnet": asdict(models.contentnet.config),
            "timbrenet": asdict(models.timbrenet.config),
            "condition_fusion": asdict(models.condition_fusion.config),
            "decoder": asdict(models.decoder.config),
            "lite_vocoder": asdict(models.lite_vocoder.config),
        },
        "models": {
            "contentnet": models.contentnet.state_dict(),
            "timbrenet": models.timbrenet.state_dict(),
            "condition_fusion": models.condition_fusion.state_dict(),
            "decoder": models.decoder.state_dict(),
            "lite_vocoder": models.lite_vocoder.state_dict(),
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
        "staged": payload["staged"],
        "models": sorted(payload["models"]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
