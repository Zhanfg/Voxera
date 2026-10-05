from __future__ import annotations

import numpy as np

from voxera.conditioning import align_frame_conditions, pitch_condition_features
from voxera.pitch import PitchTrack
from voxera.speaker import SpeakerEmbedding


def _pitch_track(frame_count: int) -> PitchTrack:
    f0 = np.full(frame_count, 220.0, dtype=np.float32)
    periodicity = np.full(frame_count, 0.9, dtype=np.float32)
    voiced = np.ones(frame_count, dtype=np.bool_)
    return PitchTrack(
        f0_hz=f0,
        periodicity=periodicity,
        voiced=voiced,
        sample_rate=16_000,
        frame_samples=640,
        hop_samples=160,
    )


def test_pitch_features_encode_log_f0_voicing_and_periodicity() -> None:
    track = _pitch_track(3)
    features = pitch_condition_features(track, reference_hz=55.0)

    assert features.shape == (3, 3)
    assert np.allclose(features[:, 0], 2.0)
    assert np.allclose(features[:, 1], 1.0)
    assert np.allclose(features[:, 2], 0.9)


def test_unvoiced_pitch_has_zero_log_f0() -> None:
    track = PitchTrack(
        f0_hz=np.array([0.0, 220.0], dtype=np.float32),
        periodicity=np.array([0.1, 0.9], dtype=np.float32),
        voiced=np.array([False, True]),
        sample_rate=16_000,
        frame_samples=640,
        hop_samples=160,
    )
    features = pitch_condition_features(track)

    assert features[0, 0] == 0.0
    assert features[0, 1] == 0.0
    assert np.isclose(features[1, 0], 2.0)


def test_alignment_crops_tail_then_applies_shared_cadence() -> None:
    content = np.arange(100 * 4, dtype=np.float32).reshape(100, 4)
    track = _pitch_track(98)
    speaker = SpeakerEmbedding(np.ones(256, dtype=np.float32))

    conditions = align_frame_conditions(content, track, speaker)

    expected_indices = np.arange(3, 98, 4)
    assert conditions.frame_count == expected_indices.size
    assert np.array_equal(conditions.content, content[expected_indices])
    assert np.allclose(conditions.pitch[:, 0], 2.0)
    assert np.isclose(conditions.frame_period_seconds, 0.04)


def test_empty_common_prefix_is_valid() -> None:
    content = np.empty((0, 256), dtype=np.float32)
    track = _pitch_track(0)
    speaker = SpeakerEmbedding(np.ones(256, dtype=np.float32))

    conditions = align_frame_conditions(content, track, speaker)

    assert conditions.frame_count == 0
    assert conditions.content.shape == (0, 256)
    assert conditions.pitch.shape == (0, 3)
