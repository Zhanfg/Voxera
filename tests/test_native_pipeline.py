from __future__ import annotations

import numpy as np

from voxera.acoustic import MelSpectrogram
from voxera.audio import AudioBuffer
from voxera.conditioning import FrameConditions
from voxera.native_pipeline import NativeModels, NativeOfflinePipeline
from voxera.speaker import SpeakerEmbedding
from voxera.vocoder import SpectralFrames, waveform_to_spectral_frames


class DummyContent:
    def encode(self, features):
        frame_count = features.shape[0]
        return np.zeros((frame_count, 256), dtype=np.float32)


class DummySpeaker:
    def encode(self, features):
        del features
        return SpeakerEmbedding(np.ones(256, dtype=np.float32))


class DummyCondition:
    def fuse(self, conditions: FrameConditions):
        return np.zeros((conditions.frame_count, 256), dtype=np.float32)


class DummyDecoder:
    def decode(self, conditions):
        return MelSpectrogram(
            np.zeros((conditions.shape[0] * 4, 80), dtype=np.float32)
        )


class DummyVocoder:
    def predict(self, mel: MelSpectrogram):
        waveform = np.zeros(mel.frame_count * 160, dtype=np.float32)
        return waveform_to_spectral_frames(waveform)


def _pipeline() -> NativeOfflinePipeline:
    return NativeOfflinePipeline(
        NativeModels(
            content=DummyContent(),
            speaker=DummySpeaker(),
            condition=DummyCondition(),
            decoder=DummyDecoder(),
            vocoder=DummyVocoder(),
        )
    )


def test_end_to_end_pipeline_produces_pcm_and_trace() -> None:
    source = AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)
    reference = AudioBuffer(np.zeros(8_000, dtype=np.float32), 16_000)

    result = _pipeline().convert_with_reference(source, reference)

    assert result.audio.sample_rate == 16_000
    assert result.trace.source_samples == 16_000
    assert result.trace.source_feature_frames == 98
    assert result.trace.pitch_frames == 97
    assert result.trace.condition_frames == 24
    assert result.trace.mel_frames == 96
    assert result.trace.spectral_frames == 96
    assert result.trace.output_samples == 96 * 160
    assert result.audio.samples.shape == (15_360,)


def test_prepared_speaker_can_be_reused() -> None:
    pipeline = _pipeline()
    reference = AudioBuffer(np.zeros(8_000, dtype=np.float32), 16_000)
    source = AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)

    speaker = pipeline.prepare_speaker(reference)
    first = pipeline.convert(source, speaker)
    second = pipeline.convert(source, speaker)

    assert np.array_equal(first.audio.samples, second.audio.samples)
    assert first.trace == second.trace


def test_pipeline_resamples_source_and_reference() -> None:
    pipeline = _pipeline()
    source = AudioBuffer(np.zeros(8_000, dtype=np.float32), 8_000)
    reference = AudioBuffer(np.zeros(4_000, dtype=np.float32), 8_000)

    result = pipeline.convert_with_reference(source, reference)

    assert result.trace.source_samples == 16_000
    assert result.audio.sample_rate == 16_000
