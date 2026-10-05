import numpy as np

from voxera.audio import AudioBuffer
from voxera.pipeline import ReferencePipeline


class DummyContentEncoder:
    def encode(self, samples, sample_rate):
        del sample_rate
        return np.stack([samples[::160], samples[::160]], axis=-1).astype(np.float32)


class DummyPitchExtractor:
    def extract(self, samples, sample_rate):
        del sample_rate
        return np.full(max(1, samples.size // 160), 120.0, dtype=np.float32)


class DummyDecoder:
    def synthesize(self, content, pitch_hz, speaker_embedding, sample_rate):
        del content, pitch_hz, speaker_embedding
        return np.zeros(sample_rate // 10, dtype=np.float32)


def test_pipeline_is_model_agnostic() -> None:
    pipeline = ReferencePipeline(DummyContentEncoder(), DummyPitchExtractor(), DummyDecoder())
    audio = AudioBuffer(np.zeros(8_000, dtype=np.float32), 8_000)
    result = pipeline.convert(audio, np.ones(64, dtype=np.float32))
    assert result.sample_rate == 16_000
    assert result.samples.shape == (1_600,)
