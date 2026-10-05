from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SpectralSynthesisConfig:
    """Streaming Fourier synthesis contract for LiteVocoder."""

    sample_rate: int = 16_000
    n_fft: int = 320
    hop_length: int = 160
    log_magnitude_min: float = -20.0
    log_magnitude_max: float = 5.0
    envelope_floor: float = 1e-8

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft <= 0 or self.n_fft % 2:
            raise ValueError("n_fft must be a positive even integer")
        if self.hop_length <= 0 or self.hop_length >= self.n_fft:
            raise ValueError("hop_length must satisfy 0 < hop_length < n_fft")
        if self.n_fft != 2 * self.hop_length:
            raise ValueError("LiteVocoder currently requires 50% overlap")
        if self.log_magnitude_min >= self.log_magnitude_max:
            raise ValueError("invalid log-magnitude clamp")
        if self.envelope_floor <= 0.0:
            raise ValueError("envelope_floor must be positive")

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def frame_period_seconds(self) -> float:
        return self.hop_length / self.sample_rate

    @property
    def algorithmic_overlap_samples(self) -> int:
        return self.n_fft - self.hop_length


@dataclass(frozen=True, slots=True)
class SpectralFrames:
    """Predicted log magnitude and phase at the vocoder frame cadence."""

    log_magnitude: FloatArray
    phase: FloatArray

    def __post_init__(self) -> None:
        log_magnitude = np.asarray(self.log_magnitude, dtype=np.float32)
        phase = np.asarray(self.phase, dtype=np.float32)
        if log_magnitude.ndim != 2 or phase.ndim != 2:
            raise ValueError("spectral arrays must be [frames, bins]")
        if log_magnitude.shape != phase.shape:
            raise ValueError("log_magnitude and phase must have the same shape")
        if not np.all(np.isfinite(log_magnitude)) or not np.all(np.isfinite(phase)):
            raise ValueError("spectral frames contain non-finite values")
        object.__setattr__(self, "log_magnitude", log_magnitude)
        object.__setattr__(self, "phase", phase)

    @property
    def frame_count(self) -> int:
        return self.log_magnitude.shape[0]

    @property
    def frequency_bins(self) -> int:
        return self.log_magnitude.shape[1]


class StreamingISTFT:
    """Explicit overlap-add iSTFT state for deterministic edge streaming.

    The neural graph predicts one spectral frame per 10 ms. This DSP stage
    converts those frames to exactly one hop of PCM per input frame. Residual
    overlap is carried as fixed-size state, so arbitrary chunking is exactly
    reproducible.
    """

    def __init__(self, config: SpectralSynthesisConfig | None = None) -> None:
        self.config = config or SpectralSynthesisConfig()
        self._window = _sine_window(self.config.n_fft)
        self._window_square = self._window * self._window
        self._accumulator = np.zeros(self.config.n_fft, dtype=np.float64)
        self._envelope = np.zeros(self.config.n_fft, dtype=np.float64)

    @property
    def pending_samples(self) -> int:
        return self.config.algorithmic_overlap_samples

    def push(self, frames: SpectralFrames) -> FloatArray:
        if frames.frequency_bins != self.config.frequency_bins:
            raise ValueError(
                f"expected {self.config.frequency_bins} bins, got {frames.frequency_bins}"
            )
        if frames.frame_count == 0:
            return np.empty(0, dtype=np.float32)

        output = np.empty(
            frames.frame_count * self.config.hop_length,
            dtype=np.float32,
        )
        cursor = 0

        for log_magnitude, phase in zip(
            frames.log_magnitude,
            frames.phase,
            strict=True,
        ):
            magnitude = np.exp(
                np.clip(
                    log_magnitude,
                    self.config.log_magnitude_min,
                    self.config.log_magnitude_max,
                )
            )
            spectrum = magnitude * (
                np.cos(phase).astype(np.float64)
                + 1j * np.sin(phase).astype(np.float64)
            )
            time_frame = np.fft.irfft(
                spectrum,
                n=self.config.n_fft,
            ).astype(np.float64, copy=False)

            self._accumulator += time_frame * self._window
            self._envelope += self._window_square

            hop = self.config.hop_length
            block = self._accumulator[:hop] / np.maximum(
                self._envelope[:hop],
                self.config.envelope_floor,
            )
            output[cursor : cursor + hop] = block.astype(np.float32)
            cursor += hop
            self._shift_state()

        return output

    def reset(self) -> None:
        self._accumulator.fill(0.0)
        self._envelope.fill(0.0)

    def _shift_state(self) -> None:
        hop = self.config.hop_length
        keep = self.config.n_fft - hop

        self._accumulator[:keep] = self._accumulator[hop:]
        self._accumulator[keep:] = 0.0
        self._envelope[:keep] = self._envelope[hop:]
        self._envelope[keep:] = 0.0


def waveform_to_spectral_frames(
    waveform: FloatArray,
    *,
    config: SpectralSynthesisConfig | None = None,
) -> SpectralFrames:
    """Create perfect-reconstruction spectral targets for testing/training.

    The waveform length must be an integer number of hops. A right tail is
    added so each emitted hop has one corresponding 50%-overlap Fourier frame.
    """

    cfg = config or SpectralSynthesisConfig()
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("waveform must be mono 1-D audio")
    if not np.all(np.isfinite(audio)):
        raise ValueError("waveform contains non-finite values")
    if audio.size % cfg.hop_length:
        raise ValueError("waveform length must be a multiple of hop_length")

    frame_count = audio.size // cfg.hop_length
    if frame_count == 0:
        empty = np.empty((0, cfg.frequency_bins), dtype=np.float32)
        return SpectralFrames(empty, empty.copy())

    padded = np.pad(
        audio,
        (0, cfg.algorithmic_overlap_samples),
        mode="constant",
    )
    window = _sine_window(cfg.n_fft)

    log_magnitude = np.empty(
        (frame_count, cfg.frequency_bins),
        dtype=np.float32,
    )
    phase = np.empty_like(log_magnitude)

    for index in range(frame_count):
        start = index * cfg.hop_length
        segment = padded[start : start + cfg.n_fft]
        spectrum = np.fft.rfft(segment * window, n=cfg.n_fft)
        log_magnitude[index] = np.log(
            np.maximum(np.abs(spectrum), np.exp(cfg.log_magnitude_min))
        ).astype(np.float32)
        phase[index] = np.angle(spectrum).astype(np.float32)

    return SpectralFrames(log_magnitude, phase)


def _sine_window(length: int) -> NDArray[np.float64]:
    indices = np.arange(length, dtype=np.float64)
    return np.sin(np.pi * (indices + 0.5) / length)
