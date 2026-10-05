from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class VocoderLossConfig:
    n_fft: int = 320
    hop_length: int = 160
    log_magnitude_weight: float = 1.0
    phase_weight: float = 0.25
    waveform_weight: float = 0.5
    eps: float = 1e-7

    def __post_init__(self) -> None:
        if self.n_fft <= 0 or self.n_fft % 2:
            raise ValueError("n_fft must be a positive even integer")
        if self.hop_length <= 0 or self.n_fft != 2 * self.hop_length:
            raise ValueError("loss currently requires 50% overlap")
        if min(
            self.log_magnitude_weight,
            self.phase_weight,
            self.waveform_weight,
        ) < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


def vocoder_reconstruction_loss(
    predicted_log_magnitude: Tensor,
    predicted_phase: Tensor,
    target_waveform: Tensor,
    config: VocoderLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Train Fourier frames against perfectly aligned waveform targets."""

    cfg = config or VocoderLossConfig()
    if predicted_log_magnitude.shape != predicted_phase.shape:
        raise ValueError("predicted magnitude and phase shapes must match")
    if predicted_log_magnitude.ndim != 3:
        raise ValueError("predicted spectra must be [batch, frames, bins]")
    if target_waveform.ndim != 2:
        raise ValueError("target_waveform must be [batch, samples]")

    expected_bins = cfg.n_fft // 2 + 1
    if predicted_log_magnitude.shape[-1] != expected_bins:
        raise ValueError(f"expected {expected_bins} frequency bins")

    frame_count = predicted_log_magnitude.shape[1]
    expected_samples = frame_count * cfg.hop_length
    if target_waveform.shape != (
        predicted_log_magnitude.shape[0],
        expected_samples,
    ):
        raise ValueError(
            f"expected target waveform shape "
            f"[{predicted_log_magnitude.shape[0]}, {expected_samples}]"
        )

    target_log_magnitude, target_phase = _spectral_targets(target_waveform, cfg)
    magnitude_loss = F.l1_loss(
        predicted_log_magnitude,
        target_log_magnitude,
    )

    magnitude_weight = torch.exp(
        torch.clamp(target_log_magnitude, max=5.0)
    )
    magnitude_weight = magnitude_weight / (
        magnitude_weight.mean(dim=-1, keepdim=True) + cfg.eps
    )
    magnitude_weight = torch.clamp(magnitude_weight, max=4.0)

    phase_distance = 1.0 - torch.cos(predicted_phase - target_phase)
    phase_loss = (phase_distance * magnitude_weight).mean()

    predicted_waveform = _synthesize(
        predicted_log_magnitude,
        predicted_phase,
        cfg,
    )
    waveform_loss = F.l1_loss(predicted_waveform, target_waveform)

    total = (
        cfg.log_magnitude_weight * magnitude_loss
        + cfg.phase_weight * phase_loss
        + cfg.waveform_weight * waveform_loss
    )
    return total, {
        "log_magnitude": magnitude_loss.detach(),
        "phase": phase_loss.detach(),
        "waveform": waveform_loss.detach(),
        "total": total.detach(),
    }


def _spectral_targets(
    waveform: Tensor,
    config: VocoderLossConfig,
) -> tuple[Tensor, Tensor]:
    frame_count = waveform.shape[1] // config.hop_length
    padded = F.pad(
        waveform,
        (0, config.n_fft - config.hop_length),
    )
    frames = padded.unfold(
        dimension=1,
        size=config.n_fft,
        step=config.hop_length,
    )
    if frames.shape[1] != frame_count:
        raise RuntimeError("unexpected target frame count")

    window = _sine_window(
        config.n_fft,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    spectrum = torch.fft.rfft(frames * window, n=config.n_fft, dim=-1)
    magnitude = torch.abs(spectrum)
    log_magnitude = torch.log(torch.clamp(magnitude, min=config.eps))
    phase = torch.angle(spectrum)
    return log_magnitude, phase


def _synthesize(
    log_magnitude: Tensor,
    phase: Tensor,
    config: VocoderLossConfig,
) -> Tensor:
    magnitude = torch.exp(torch.clamp(log_magnitude, min=-20.0, max=5.0))
    spectrum = torch.polar(magnitude, phase)
    frames = torch.fft.irfft(spectrum, n=config.n_fft, dim=-1)

    window = _sine_window(
        config.n_fft,
        device=frames.device,
        dtype=frames.dtype,
    )
    weighted = frames * window
    window_square = window.square()

    batch, frame_count, _ = weighted.shape
    output_length = frame_count * config.hop_length
    accumulation = torch.zeros(
        batch,
        output_length + config.n_fft - config.hop_length,
        device=frames.device,
        dtype=frames.dtype,
    )
    envelope = torch.zeros_like(accumulation)

    for index in range(frame_count):
        start = index * config.hop_length
        end = start + config.n_fft
        accumulation[:, start:end] = accumulation[:, start:end] + weighted[:, index]
        envelope[:, start:end] = envelope[:, start:end] + window_square

    audio = accumulation[:, :output_length] / torch.clamp(
        envelope[:, :output_length],
        min=config.eps,
    )
    return audio


def _sine_window(
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    indices = torch.arange(length, device=device, dtype=dtype)
    return torch.sin(pi * (indices + 0.5) / length)
