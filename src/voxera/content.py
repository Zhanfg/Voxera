from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class ContentCadence:
    """Map dense 10 ms ContentNet frames onto a lower-rate VC cadence."""

    stride: int = 4
    offset: int = 3

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not 0 <= self.offset < self.stride:
            raise ValueError("offset must satisfy 0 <= offset < stride")

    def select(self, features: FloatArray) -> FloatArray:
        values = _validate_features(features)
        return values[self.offset :: self.stride].copy()


class StreamingCadenceSelector:
    """Chunk-safe selector matching :meth:`ContentCadence.select` exactly."""

    def __init__(self, cadence: ContentCadence | None = None) -> None:
        self.cadence = cadence or ContentCadence()
        self._frames_seen = 0

    @property
    def phase(self) -> int:
        return self._frames_seen % self.cadence.stride

    def push(self, features: FloatArray) -> FloatArray:
        values = _validate_features(features)
        if values.shape[0] == 0:
            return values.copy()

        local_indices = np.arange(values.shape[0], dtype=np.int64)
        global_phase = (self._frames_seen + local_indices) % self.cadence.stride
        mask = global_phase == self.cadence.offset
        selected = values[mask].copy()
        self._frames_seen += values.shape[0]
        return selected

    def reset(self) -> None:
        self._frames_seen = 0


def _validate_features(features: FloatArray) -> FloatArray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("content features must be a 2-D time-major array")
    if not np.all(np.isfinite(values)):
        raise ValueError("content features contain non-finite values")
    return values
