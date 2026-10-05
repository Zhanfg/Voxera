from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]


class ContentEncoder(Protocol):
    def encode(self, samples: FloatArray, sample_rate: int) -> FloatArray:
        """Return time-major linguistic/content features."""
        ...


class PitchExtractor(Protocol):
    def extract(self, samples: FloatArray, sample_rate: int) -> FloatArray:
        """Return the F0 contour in Hz; unvoiced frames should be zero."""
        ...


class VoiceDecoder(Protocol):
    def synthesize(
        self,
        content: FloatArray,
        pitch_hz: FloatArray,
        speaker_embedding: FloatArray,
        sample_rate: int,
    ) -> FloatArray:
        """Synthesize mono float32 PCM from model conditions."""
        ...


@dataclass(frozen=True, slots=True)
class IntermediateFeatures:
    content: FloatArray
    pitch_hz: FloatArray
