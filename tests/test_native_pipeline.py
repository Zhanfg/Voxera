from __future__ import annotations

import numpy as np
import pytest

from voxera.acoustic import MelSpectrogram
from voxera.audio import AudioBuffer
from voxera.conditioning import FrameConditions
from voxera.native_pipeline import NativeModels, NativeOfflinePipeline
from voxera.prosody import ProsodyEmbedding
from voxera.speaker import SpeakerEmbedding
from voxera.vocoder import waveform_to_spectral_frames


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


class DummyProsody:
    def encode(self, features, pitch):
        assert features.shape[0] == pitch.shape[0]
        frames = np.arange(3, features.shape[0], 4).size
        return ProsodyEmbedding(
            local=np.zeros((frames, 32), dtype=np.float32),
            global_style=np.zeros(16, dtype=np.float32),
        )


class BadProsody:
    def encode(self, features, pitch):
        assert features.shape[0] == pitch.shape[0]
        frames = max(0, np.arange(3, features.shape[0], 4).size - 1)
        return ProsodyEmbedding(
            local=np.zeros((frames, 32), dtype=np.float32),
            global_style=np.zeros(16, dtype=np.float32),
        )


class DummyProsodyCondition:
    def __init__(self):
        self.called = False

    def apply(self, conditions, prosody):
        self.called = True
        assert conditions.shape[0] == prosody.frame_count
        return np.asarray(conditions, dtype=np.float32).copy()


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


def test_end_to_end_pipeline_preserves_exact_duration() -> None:
    source = AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)
    reference = AudioBuffer(np.zeros(8_000, dtype=np.float32), 16_000)

    result = _pipeline().convert_with_reference(source, reference)

    assert result.audio.sample_rate == 16_000
    assert result.trace.source_samples == 16_000
    assert result.trace.analysis_samples == 16_480
    assert result.trace.source_feature_frames == 101
    assert result.trace.pitch_frames == 100
    assert result.trace.condition_frames == 25
    assert result.trace.mel_frames == 100
    assert result.trace.spectral_frames == 100
    assert result.trace.generated_samples == 16_000
    assert result.trace.output_samples == 16_000
    assert result.audio.samples.shape == (16_000,)


@pytest.mark.parametrize("sample_count", [1, 159, 160, 161, 15_999, 16_001, 23_777])
def test_arbitrary_source_lengths_are_preserved(sample_count: int) -> None:
    pipeline = _pipeline()
    source = AudioBuffer(np.zeros(sample_count, dtype=np.float32), 16_000)
    speaker = SpeakerEmbedding(np.ones(256, dtype=np.float32))

    result = pipeline.convert(source, speaker)

    assert result.trace.source_samples == sample_count
    assert result.trace.output_samples == sample_count
    assert result.audio.samples.shape == (sample_count,)
    assert result.trace.generated_samples >= sample_count
    assert result.trace.mel_frames % 4 == 0


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
    assert result.trace.output_samples == 16_000
    assert result.audio.sample_rate == 16_000


def test_empty_source_is_rejected() -> None:
    pipeline = _pipeline()
    source = AudioBuffer(np.empty(0, dtype=np.float32), 16_000)
    speaker = SpeakerEmbedding(np.ones(256, dtype=np.float32))

    with pytest.raises(ValueError, match="at least one sample"):
        pipeline.convert(source, speaker)



def test_optional_prosody_path_is_applied_without_duration_change() -> None:
    conditioner = DummyProsodyCondition()
    pipeline = NativeOfflinePipeline(
        NativeModels(
            content=DummyContent(),
            speaker=DummySpeaker(),
            condition=DummyCondition(),
            decoder=DummyDecoder(),
            vocoder=DummyVocoder(),
            prosody=DummyProsody(),
            prosody_condition=conditioner,
        )
    )
    source = AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)
    speaker = SpeakerEmbedding(np.ones(256, dtype=np.float32))

    result = pipeline.convert(source, speaker)

    assert conditioner.called
    assert result.audio.samples.shape == (16_000,)
    assert result.trace.condition_frames == 25


def test_prosody_models_must_be_configured_as_a_pair() -> None:
    with pytest.raises(ValueError, match="provided together"):
        NativeModels(
            content=DummyContent(),
            speaker=DummySpeaker(),
            condition=DummyCondition(),
            decoder=DummyDecoder(),
            vocoder=DummyVocoder(),
            prosody=DummyProsody(),
        )


def test_prosody_frame_mismatch_is_rejected() -> None:
    pipeline = NativeOfflinePipeline(
        NativeModels(
            content=DummyContent(),
            speaker=DummySpeaker(),
            condition=DummyCondition(),
            decoder=DummyDecoder(),
            vocoder=DummyVocoder(),
            prosody=BadProsody(),
            prosody_condition=DummyProsodyCondition(),
        )
    )
    source = AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)
    speaker = SpeakerEmbedding(np.ones(256, dtype=np.float32))

    with pytest.raises(ValueError, match="one local embedding"):
        pipeline.convert(source, speaker)
