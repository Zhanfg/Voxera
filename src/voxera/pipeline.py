from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .audio import AudioBuffer, resample_linear
from .config import AudioConfig
from .interfaces import ContentEncoder, IntermediateFeatures, PitchExtractor, VoiceDecoder

FloatArray = NDArray[np.float32]


class ReferencePipeline:
    """Model-agnostic WAV-to-WAV orchestration for the first Voxera milestone."""

    def __init__(
        self,
        content_encoder: ContentEncoder,
        pitch_extractor: PitchExtractor,
        decoder: VoiceDecoder,
        config: AudioConfig | None = None,
    ) -> None:
        self.config = config or AudioConfig()
        self.content_encoder = content_encoder
        self.pitch_extractor = pitch_extractor
        self.decoder = decoder

    def analyze(self, audio: AudioBuffer) -> tuple[AudioBuffer, IntermediateFeatures]:
        prepared = resample_linear(audio, self.config.sample_rate)
        content = _as_finite_float32(
            self.content_encoder.encode(prepared.samples, prepared.sample_rate), "content"
        )
        pitch_hz = _as_finite_float32(
            self.pitch_extractor.extract(prepared.samples, prepared.sample_rate), "pitch_hz"
        )
        if content.ndim != 2:
            raise ValueError("content features must be a 2-D time-major array")
        if pitch_hz.ndim != 1:
            raise ValueError("pitch_hz must be a 1-D array")
        if np.any(pitch_hz < 0):
            raise ValueError("pitch_hz must be non-negative")
        return prepared, IntermediateFeatures(content=content, pitch_hz=pitch_hz)

    def convert(self, audio: AudioBuffer, speaker_embedding: FloatArray) -> AudioBuffer:
        prepared, features = self.analyze(audio)
        speaker = _as_finite_float32(speaker_embedding, "speaker_embedding")
        if speaker.ndim != 1 or speaker.size == 0:
            raise ValueError("speaker_embedding must be a non-empty 1-D array")

        output = _as_finite_float32(
            self.decoder.synthesize(
                features.content,
                features.pitch_hz,
                speaker,
                prepared.sample_rate,
            ),
            "decoder output",
        )
        if output.ndim != 1:
            raise ValueError("decoder output must be mono 1-D PCM")
        return AudioBuffer(output, prepared.sample_rate)


def _as_finite_float32(value: FloatArray, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array
