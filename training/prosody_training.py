from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from training.prosody_loss import prosody_descriptor_loss
from training.prosodynet import ProsodyNet
from voxera.audio import load_wav, resample_linear
from voxera.conditioning import pitch_condition_features
from voxera.features import LogMelFrontend
from voxera.pitch import YinPitchExtractor
from voxera.prosody import NativeProsodyExtractor


@dataclass(frozen=True, slots=True)
class ProsodyTrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_accumulation: int = 4
    max_grad_norm: float = 5.0
    max_dense_frames: int = 1_200
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
        if self.max_dense_frames < 8:
            raise ValueError("max_dense_frames must be at least 8")


@dataclass(frozen=True, slots=True)
class ProsodyTrainSummary:
    epoch: int
    files: int
    optimizer_steps: int
    mean_loss: float
    checkpoint: Path


def train_prosodynet(
    wav_dir: Path,
    output_dir: Path,
    *,
    epochs: int,
    device: torch.device,
    config: ProsodyTrainConfig | None = None,
    max_files: int | None = None,
) -> list[ProsodyTrainSummary]:
    """Bootstrap ProsodyNet from native acoustic style pseudo-targets.

    This stage requires only ordinary WAV files. It does not depend on MeanVC2
    or another large teacher model, so it can be developed and debugged before
    the main VC dataset/GPU training run exists.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    cfg = config or ProsodyTrainConfig()
    wavs = sorted(
        path.resolve()
        for path in wav_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    if max_files is not None:
        if max_files <= 0:
            raise ValueError("max_files must be positive")
        wavs = wavs[:max_files]
    if not wavs:
        raise ValueError(f"no WAV files found in {wav_dir}")

    model = ProsodyNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    frontend = LogMelFrontend()
    pitch_extractor = YinPitchExtractor()
    prosody_extractor = NativeProsodyExtractor()
    rng = random.Random(cfg.seed)
    dataset_sha = _dataset_fingerprint(wavs)
    global_step = 0
    summaries: list[ProsodyTrainSummary] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(wavs)
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        seen = 0
        accumulated = 0
        optimizer_steps = 0

        for path in order:
            sample = _training_sample(
                path,
                frontend,
                pitch_extractor,
                prosody_extractor,
                device,
                cfg.max_dense_frames,
                rng,
            )
            if sample is None:
                continue

            acoustic, pitch, local_target, global_target = sample
            (
                local_embedding,
                global_embedding,
                local_prediction,
                global_prediction,
            ) = model(
                acoustic.unsqueeze(0),
                pitch.unsqueeze(0),
            )

            local_prediction_40 = local_prediction[:, 3::4]
            local_embedding_40 = local_embedding[:, 3::4]
            common = min(local_prediction_40.shape[1], local_target.shape[0])
            if common <= 0:
                continue

            loss, _ = prosody_descriptor_loss(
                local_embedding_40[:, :common],
                global_embedding,
                local_prediction_40[:, :common],
                global_prediction,
                local_target[:common].unsqueeze(0),
                global_target.unsqueeze(0),
            )
            loss.backward()

            total_loss += float(loss.detach())
            seen += 1
            accumulated += 1
            global_step += 1

            if accumulated == cfg.gradient_accumulation:
                _average_gradients(model.parameters(), accumulated)
                clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulated = 0

        if accumulated:
            _average_gradients(model.parameters(), accumulated)
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if seen == 0:
            raise RuntimeError("no WAV produced a valid prosody training sample")

        checkpoint = output_dir / f"prosodynet-epoch{epoch:03d}.pt"
        _save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            config=cfg,
            wav_dir=wav_dir,
            dataset_sha=dataset_sha,
        )
        summaries.append(
            ProsodyTrainSummary(
                epoch=epoch,
                files=seen,
                optimizer_steps=optimizer_steps,
                mean_loss=total_loss / seen,
                checkpoint=checkpoint,
            )
        )

    return summaries


def _training_sample(
    path: Path,
    frontend: LogMelFrontend,
    pitch_extractor: YinPitchExtractor,
    prosody_extractor: NativeProsodyExtractor,
    device: torch.device,
    max_dense_frames: int,
    rng: random.Random,
):
    audio = resample_linear(load_wav(path), 16_000)
    features = frontend.extract(audio.samples).values
    pitch_track = pitch_extractor.extract(audio.samples)
    dense_pitch = pitch_condition_features(pitch_track)

    common = min(features.shape[0], dense_pitch.shape[0])
    if common < 4:
        return None

    features = features[:common]
    dense_pitch = dense_pitch[:common]

    if common > max_dense_frames:
        aligned_limit = max_dense_frames - (max_dense_frames % 4)
        start_limit = common - aligned_limit
        start = rng.randrange(0, start_limit + 1)
        start -= start % 4
        end = start + aligned_limit
        features = features[start:end]
        dense_pitch = dense_pitch[start:end]

        sample_start = start * pitch_track.hop_samples
        sample_end = (
            (end - 1) * pitch_track.hop_samples
            + pitch_track.frame_samples
        )
        cropped_audio = audio.samples[sample_start:sample_end]
        cropped_track = pitch_extractor.extract(cropped_audio)
        target = prosody_extractor.extract(cropped_audio, cropped_track)
    else:
        target = prosody_extractor.extract(audio.samples, pitch_track)

    if target.frame_count == 0:
        return None

    return (
        torch.from_numpy(features.copy()).to(device=device, dtype=torch.float32),
        torch.from_numpy(dense_pitch.copy()).to(device=device, dtype=torch.float32),
        torch.from_numpy(target.local.copy()).to(device=device, dtype=torch.float32),
        torch.from_numpy(target.global_style.copy()).to(
            device=device,
            dtype=torch.float32,
        ),
    )


def _average_gradients(parameters, accumulated: int) -> None:
    scale = 1.0 / accumulated
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def _save_checkpoint(
    path: Path,
    *,
    model: ProsodyNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: ProsodyTrainConfig,
    wav_dir: Path,
    dataset_sha: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "component": "prosodynet",
        "epoch": epoch,
        "global_step": global_step,
        "dataset": {
            "wav_dir": str(wav_dir.resolve()),
            "sha256": dataset_sha,
        },
        "model_config": asdict(model.config),
        "train_config": asdict(config),
        "model": model.state_dict(),
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
        "dataset": payload["dataset"],
        "model_config": payload["model_config"],
        "train_config": payload["train_config"],
    }


def _dataset_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
