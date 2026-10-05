from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SpeakerEmbedding:
    """Normalized global target-speaker representation."""

    values: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("speaker embedding must be a non-empty 1-D array")
        if not np.all(np.isfinite(values)):
            raise ValueError("speaker embedding contains non-finite values")

        norm = float(np.linalg.norm(values))
        if norm <= 0.0:
            raise ValueError("speaker embedding must have non-zero norm")
        object.__setattr__(self, "values", (values / norm).astype(np.float32))

    @property
    def dimension(self) -> int:
        return self.values.size


def cosine_similarity(left: SpeakerEmbedding, right: SpeakerEmbedding) -> float:
    if left.dimension != right.dimension:
        raise ValueError("speaker embeddings must have the same dimension")
    return float(np.dot(left.values, right.values))
