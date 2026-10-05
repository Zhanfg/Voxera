from __future__ import annotations

import numpy as np
import pytest

from voxera.acoustic import MelConfig, MelSpectrogram, expected_mel_frames


def test_mel_spectrogram_contract() -> None:
    config = MelConfig()
    mel = MelSpectrogram(np.zeros((100, 80), dtype=np.float32), config)

    assert mel.frame_count == 100
    assert np.isclose(mel.duration_seconds, 1.0)
    assert config.hop_samples == 160


def test_expected_mel_frames_matches_40_to_10_ms_expansion() -> None:
    assert expected_mel_frames(37) == 148
    assert expected_mel_frames(0) == 0


def test_wrong_mel_dimension_is_rejected() -> None:
    with pytest.raises(ValueError):
        MelSpectrogram(np.zeros((10, 79), dtype=np.float32))


@pytest.mark.parametrize(
    ("condition_frames", "upsample_factor"),
    [
        (-1, 4),
        (1, 0),
    ],
)
def test_invalid_frame_count_is_rejected(
    condition_frames: int,
    upsample_factor: int,
) -> None:
    with pytest.raises(ValueError):
        expected_mel_frames(
            condition_frames,
            upsample_factor=upsample_factor,
        )
