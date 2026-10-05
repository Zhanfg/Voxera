from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class PitchConfig:
    """Configuration for Voxera's dependency-light speech pitch extractor."""

    sample_rate: int = 16_000
    frame_ms: float = 40.0
    hop_ms: float = 10.0
    f_min: float = 50.0
    f_max: float = 550.0
    yin_threshold: float = 0.15
    periodicity_threshold: float = 0.55
    energy_floor: float = 1e-5

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms must not exceed frame_ms")
        if not 0.0 < self.f_min < self.f_max < self.sample_rate / 2:
            raise ValueError("expected 0 < f_min < f_max < Nyquist")
        if not 0.0 < self.yin_threshold < 1.0:
            raise ValueError("yin_threshold must be in (0, 1)")
        if not 0.0 <= self.periodicity_threshold <= 1.0:
            raise ValueError("periodicity_threshold must be in [0, 1]")
        if self.energy_floor <= 0.0:
            raise ValueError("energy_floor must be positive")
        if self.max_lag >= self.frame_samples:
            raise ValueError("frame is too short for f_min")

    @property
    def frame_samples(self) -> int:
        return round(self.sample_rate * self.frame_ms / 1000.0)

    @property
    def hop_samples(self) -> int:
        return round(self.sample_rate * self.hop_ms / 1000.0)

    @property
    def min_lag(self) -> int:
        return max(1, int(self.sample_rate / self.f_max))

    @property
    def max_lag(self) -> int:
        return int(np.ceil(self.sample_rate / self.f_min))


@dataclass(frozen=True, slots=True)
class PitchTrack:
    """Frame-aligned pitch and voicing evidence."""

    f0_hz: FloatArray
    periodicity: FloatArray
    voiced: NDArray[np.bool_]
    sample_rate: int
    frame_samples: int
    hop_samples: int

    def __post_init__(self) -> None:
        f0_hz = np.asarray(self.f0_hz, dtype=np.float32)
        periodicity = np.asarray(self.periodicity, dtype=np.float32)
        voiced = np.asarray(self.voiced, dtype=np.bool_)

        if f0_hz.ndim != 1 or periodicity.ndim != 1 or voiced.ndim != 1:
            raise ValueError("pitch arrays must be one-dimensional")
        if not (f0_hz.shape == periodicity.shape == voiced.shape):
            raise ValueError("pitch arrays must have matching shapes")
        if not np.all(np.isfinite(f0_hz)) or not np.all(np.isfinite(periodicity)):
            raise ValueError("pitch track contains non-finite values")
        if np.any(f0_hz < 0.0):
            raise ValueError("f0_hz must be non-negative")
        if np.any((periodicity < 0.0) | (periodicity > 1.0)):
            raise ValueError("periodicity must be in [0, 1]")

        object.__setattr__(self, "f0_hz", f0_hz)
        object.__setattr__(self, "periodicity", periodicity)
        object.__setattr__(self, "voiced", voiced)

    @property
    def frame_count(self) -> int:
        return self.f0_hz.size


class YinPitchExtractor:
    """Small deterministic YIN-style speech F0 extractor.

    This serves as Voxera's native low-power pitch baseline and fallback. The
    neural PitchNet can later refine or replace it without changing PitchTrack.
    """

    def __init__(self, config: PitchConfig | None = None) -> None:
        self.config = config or PitchConfig()

    def extract(self, samples: FloatArray) -> PitchTrack:
        audio = _validate_audio(samples)
        config = self.config
        frame_count = _frame_count(audio.size, config.frame_samples, config.hop_samples)

        if frame_count == 0:
            return self._empty_track()

        f0 = np.zeros(frame_count, dtype=np.float32)
        periodicity = np.zeros(frame_count, dtype=np.float32)
        voiced = np.zeros(frame_count, dtype=np.bool_)

        for index in range(frame_count):
            start = index * config.hop_samples
            frame = audio[start : start + config.frame_samples]
            estimate, confidence = self._estimate_frame(frame)
            f0[index] = estimate
            periodicity[index] = confidence
            voiced[index] = estimate > 0.0

        return PitchTrack(
            f0_hz=f0,
            periodicity=periodicity,
            voiced=voiced,
            sample_rate=config.sample_rate,
            frame_samples=config.frame_samples,
            hop_samples=config.hop_samples,
        )

    def _estimate_frame(self, frame: FloatArray) -> tuple[float, float]:
        config = self.config
        centered = frame - float(np.mean(frame))
        rms = float(np.sqrt(np.mean(centered * centered)))
        if rms < config.energy_floor:
            return 0.0, 0.0

        analysis = centered
        max_lag = config.max_lag
        difference = np.zeros(max_lag + 1, dtype=np.float64)

        for lag in range(1, max_lag + 1):
            delta = analysis[:-lag] - analysis[lag:]
            difference[lag] = float(np.dot(delta, delta))

        cmndf = np.ones(max_lag + 1, dtype=np.float64)
        cumulative = 0.0
        for lag in range(1, max_lag + 1):
            cumulative += difference[lag]
            if cumulative > 0.0:
                cmndf[lag] = difference[lag] * lag / cumulative

        lag = _select_lag(
            cmndf,
            config.min_lag,
            config.max_lag,
            config.yin_threshold,
        )
        refined_lag = _parabolic_minimum(cmndf, lag)
        confidence = float(np.clip(1.0 - cmndf[lag], 0.0, 1.0))

        if confidence < config.periodicity_threshold or refined_lag <= 0.0:
            return 0.0, confidence

        frequency = config.sample_rate / refined_lag
        if not config.f_min <= frequency <= config.f_max:
            return 0.0, confidence
        return float(frequency), confidence

    def _empty_track(self) -> PitchTrack:
        empty = np.empty(0, dtype=np.float32)
        return PitchTrack(
            f0_hz=empty,
            periodicity=empty.copy(),
            voiced=np.empty(0, dtype=np.bool_),
            sample_rate=self.config.sample_rate,
            frame_samples=self.config.frame_samples,
            hop_samples=self.config.hop_samples,
        )


class StreamingYinPitchExtractor:
    """Chunk-safe streaming wrapper for :class:`YinPitchExtractor`."""

    def __init__(self, config: PitchConfig | None = None) -> None:
        self.extractor = YinPitchExtractor(config)
        self._buffer = np.empty(0, dtype=np.float32)

    @property
    def config(self) -> PitchConfig:
        return self.extractor.config

    @property
    def buffered_samples(self) -> int:
        return self._buffer.size

    def push(self, samples: FloatArray) -> PitchTrack:
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
            return self.extractor._empty_track()

        final_frame_end = (
            (frame_count - 1) * config.hop_samples + config.frame_samples
        )
        ready = self._buffer[:final_frame_end]
        track = self.extractor.extract(ready)

        consumed_starts = frame_count * config.hop_samples
        self._buffer = self._buffer[consumed_starts:].copy()
        return track

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.float32)


def _validate_audio(samples: FloatArray) -> FloatArray:
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("pitch extractor expects mono 1-D audio")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains non-finite values")
    return audio


def _frame_count(sample_count: int, frame_samples: int, hop_samples: int) -> int:
    if sample_count < frame_samples:
        return 0
    return 1 + (sample_count - frame_samples) // hop_samples


def _select_lag(
    cmndf: NDArray[np.float64],
    min_lag: int,
    max_lag: int,
    threshold: float,
) -> int:
    candidate = min_lag
    while candidate <= max_lag:
        if cmndf[candidate] < threshold:
            while candidate + 1 <= max_lag and cmndf[candidate + 1] < cmndf[candidate]:
                candidate += 1
            return candidate
        candidate += 1

    region = cmndf[min_lag : max_lag + 1]
    return min_lag + int(np.argmin(region))


def _parabolic_minimum(values: NDArray[np.float64], index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return float(index)

    left = values[index - 1]
    center = values[index]
    right = values[index + 1]
    denominator = left - 2.0 * center + right
    if abs(denominator) < np.finfo(np.float64).eps:
        return float(index)

    correction = 0.5 * (left - right) / denominator
    return float(index + np.clip(correction, -1.0, 1.0))
