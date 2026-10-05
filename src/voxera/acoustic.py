from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class MelConfig:
    """Timebase and dimensionality for Voxera decoder mel output."""

    sample_rate: int = 16_000
    hop_ms: float = 10.0
    n_mels: int = 80

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_ms <= 0.0:
            raise ValueError("hop_ms must be positive")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")

    @property
    def hop_samples(self) -> int:
        return round(self.sample_rate * self.hop_ms / 1000.0)


@dataclass(frozen=True, slots=True)
class MelSpectrogram:
    """Time-major log-mel spectrogram emitted by the acoustic decoder."""

    values: FloatArray
    config: MelConfig = MelConfig()

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("mel values must be a 2-D time-major array")
        if values.shape[1] != self.config.n_mels:
            raise ValueError(
                f"expected {self.config.n_mels} mel bins, got {values.shape[1]}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("mel values contain non-finite values")
        object.__setattr__(self, "values", values)

    @property
    def frame_count(self) -> int:
        return self.values.shape[0]

    @property
    def duration_seconds(self) -> float:
        return self.frame_count * self.config.hop_samples / self.config.sample_rate


def expected_mel_frames(
    condition_frames: int,
    *,
    upsample_factor: int = 4,
) -> int:
    if condition_frames < 0:
        raise ValueError("condition_frames must be non-negative")
    if upsample_factor <= 0:
        raise ValueError("upsample_factor must be positive")
    return condition_frames * upsample_factor
