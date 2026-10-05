from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FbankConfig:
    """Configuration for Voxera's native acoustic frontend."""

    sample_rate: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float | None = None
    preemphasis: float = 0.97
    eps: float = 1e-10

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms must not exceed frame_ms")
        if self.n_fft <= 0 or self.n_fft < self.frame_samples:
            raise ValueError("n_fft must be at least one frame long")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if not 0.0 <= self.preemphasis < 1.0:
            raise ValueError("preemphasis must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if not 0.0 <= self.f_min < self.max_frequency <= self.nyquist:
            raise ValueError("expected 0 <= f_min < f_max <= Nyquist")

    @property
    def frame_samples(self) -> int:
        return round(self.sample_rate * self.frame_ms / 1000.0)

    @property
    def hop_samples(self) -> int:
        return round(self.sample_rate * self.hop_ms / 1000.0)

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0

    @property
    def max_frequency(self) -> float:
        return self.nyquist if self.f_max is None else self.f_max


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    """Time-major acoustic features with an explicit timebase."""

    values: FloatArray
    sample_rate: int
    frame_samples: int
    hop_samples: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("feature values must be a 2-D time-major array")
        if not np.all(np.isfinite(values)):
            raise ValueError("feature values contain non-finite values")
        object.__setattr__(self, "values", values)

    @property
    def frame_count(self) -> int:
        return self.values.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.values.shape[1]

    @property
    def hop_seconds(self) -> float:
        return self.hop_samples / self.sample_rate


class LogMelFrontend:
    """Dependency-light 80-bin log-mel frontend for ContentNet.

    The implementation is intentionally explicit so the same math can later be
    ported to the native mobile runtime. Frames use a Kaldi-style 25 ms / 10 ms
    cadence by default, frame-local pre-emphasis, and a Povey-like window.
    """

    def __init__(self, config: FbankConfig | None = None) -> None:
        self.config = config or FbankConfig()
        self._window = _povey_window(self.config.frame_samples)
        self._mel_filters = _mel_filterbank(self.config)

    def extract(self, samples: FloatArray) -> FeatureBatch:
        audio = _validate_audio(samples)
        config = self.config
        frame_count = _frame_count(audio.size, config.frame_samples, config.hop_samples)
        if frame_count == 0:
            return self._batch(np.empty((0, config.n_mels), dtype=np.float32))

        frames = np.empty((frame_count, config.frame_samples), dtype=np.float32)
        for index in range(frame_count):
            start = index * config.hop_samples
            frames[index] = audio[start : start + config.frame_samples]

        emphasized = _preemphasize_frames(frames, config.preemphasis)
        windowed = emphasized * self._window
        spectrum = np.fft.rfft(windowed, n=config.n_fft, axis=1)
        power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(
            np.float32
        )
        mel_energy = power @ self._mel_filters.T
        features = np.log(np.maximum(mel_energy, config.eps)).astype(np.float32)
        return self._batch(features)

    def _batch(self, values: FloatArray) -> FeatureBatch:
        return FeatureBatch(
            values=values,
            sample_rate=self.config.sample_rate,
            frame_samples=self.config.frame_samples,
            hop_samples=self.config.hop_samples,
        )


class StreamingLogMelFrontend:
    """Chunk-safe streaming wrapper around :class:`LogMelFrontend`.

    Only samples that can contribute to a future complete frame are retained.
    With snipped-edge framing, concatenating all returned batches is equivalent
    to one offline extraction over the concatenated input.
    """

    def __init__(self, config: FbankConfig | None = None) -> None:
        self.frontend = LogMelFrontend(config)
        self._buffer = np.empty(0, dtype=np.float32)

    @property
    def config(self) -> FbankConfig:
        return self.frontend.config

    @property
    def buffered_samples(self) -> int:
        return self._buffer.size

    def push(self, samples: FloatArray) -> FeatureBatch:
        chunk = _validate_audio(samples)
        if chunk.size:
            self._buffer = np.concatenate((self._buffer, chunk))

        config = self.config
        frame_count = _frame_count(
            self._buffer.size,
            config.frame_samples,
            config.hop_samples,
        )
        if frame_count == 0:
            return self.frontend._batch(
                np.empty((0, config.n_mels), dtype=np.float32)
            )

        final_frame_end = (
            (frame_count - 1) * config.hop_samples + config.frame_samples
        )
        ready = self._buffer[:final_frame_end]
        batch = self.frontend.extract(ready)

        consumed_starts = frame_count * config.hop_samples
        self._buffer = self._buffer[consumed_starts:].copy()
        return batch

    def flush(self) -> FeatureBatch:
        """Discard an incomplete tail and reset the streaming state."""

        self._buffer = np.empty(0, dtype=np.float32)
        return self.frontend._batch(
            np.empty((0, self.config.n_mels), dtype=np.float32)
        )

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)


def _validate_audio(samples: FloatArray) -> FloatArray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("acoustic frontend expects mono 1-D audio")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains non-finite values")
    return audio


def _frame_count(sample_count: int, frame_samples: int, hop_samples: int) -> int:
    if sample_count < frame_samples:
        return 0
    return 1 + (sample_count - frame_samples) // hop_samples


def _preemphasize_frames(frames: FloatArray, coefficient: float) -> FloatArray:
    if coefficient == 0.0:
        return frames.copy()

    output = frames.copy()
    output[:, 1:] = frames[:, 1:] - coefficient * frames[:, :-1]
    output[:, 0] = frames[:, 0] * (1.0 - coefficient)
    return output


def _povey_window(length: int) -> FloatArray:
    # Kaldi's "povey" window is a Hann window raised to 0.85.
    return np.power(np.hanning(length), 0.85).astype(np.float32)


def _hz_to_mel(frequency_hz: FloatArray | float) -> NDArray[np.float64]:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: FloatArray | float) -> NDArray[np.float64]:
    mel_value = np.asarray(mel, dtype=np.float64)
    return 700.0 * (np.power(10.0, mel_value / 2595.0) - 1.0)


def _mel_filterbank(config: FbankConfig) -> FloatArray:
    fft_frequencies = np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)
    mel_points = np.linspace(
        _hz_to_mel(config.f_min),
        _hz_to_mel(config.max_frequency),
        config.n_mels + 2,
    )
    hz_points = _mel_to_hz(mel_points)

    filters = np.zeros((config.n_mels, fft_frequencies.size), dtype=np.float32)
    for index in range(config.n_mels):
        left, center, right = hz_points[index : index + 3]
        rising = (fft_frequencies - left) / max(center - left, np.finfo(float).eps)
        falling = (right - fft_frequencies) / max(
            right - center,
            np.finfo(float).eps,
        )
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))

    return filters
