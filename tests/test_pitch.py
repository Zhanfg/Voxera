from __future__ import annotations

import numpy as np
import pytest

from voxera.pitch import PitchConfig, StreamingYinPitchExtractor, YinPitchExtractor


def _sine(frequency: float, seconds: float = 1.0, sample_rate: int = 16_000) -> np.ndarray:
    time = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    return np.sin(2.0 * np.pi * frequency * time).astype(np.float32)


@pytest.mark.parametrize("frequency", [80.0, 110.0, 220.0, 440.0])
def test_sine_frequency_is_recovered(frequency: float) -> None:
    track = YinPitchExtractor().extract(_sine(frequency))

    voiced = track.f0_hz[track.voiced]
    assert voiced.size > 0
    assert abs(float(np.median(voiced)) - frequency) < 1.5
    assert float(np.median(track.periodicity[track.voiced])) > 0.8


def test_silence_is_unvoiced() -> None:
    track = YinPitchExtractor().extract(np.zeros(16_000, dtype=np.float32))

    assert track.frame_count > 0
    assert not np.any(track.voiced)
    assert np.all(track.f0_hz == 0.0)
    assert np.all(track.periodicity == 0.0)


def test_streaming_matches_offline_for_irregular_chunks() -> None:
    samples = np.concatenate(
        (
            _sine(120.0, 0.7),
            np.zeros(2_000, dtype=np.float32),
            _sine(230.0, 0.9),
        )
    )
    offline = YinPitchExtractor().extract(samples)
    streaming = StreamingYinPitchExtractor()

    chunks = (1, 17, 333, 2_048, 79, 4_096, 11, 7_000)
    f0_parts = []
    periodicity_parts = []
    voiced_parts = []
    cursor = 0

    for size in chunks:
        end = min(samples.size, cursor + size)
        track = streaming.push(samples[cursor:end])
        if track.frame_count:
            f0_parts.append(track.f0_hz)
            periodicity_parts.append(track.periodicity)
            voiced_parts.append(track.voiced)
        cursor = end
        if cursor == samples.size:
            break

    if cursor < samples.size:
        track = streaming.push(samples[cursor:])
        f0_parts.append(track.f0_hz)
        periodicity_parts.append(track.periodicity)
        voiced_parts.append(track.voiced)

    f0 = np.concatenate(f0_parts)
    periodicity = np.concatenate(periodicity_parts)
    voiced = np.concatenate(voiced_parts)

    assert np.array_equal(voiced, offline.voiced)
    assert np.allclose(f0, offline.f0_hz, atol=1e-5)
    assert np.allclose(periodicity, offline.periodicity, atol=1e-6)


def test_short_input_yields_empty_track() -> None:
    config = PitchConfig()
    track = YinPitchExtractor(config).extract(
        np.zeros(config.frame_samples - 1, dtype=np.float32)
    )
    assert track.frame_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"f_min": 0.0},
        {"f_min": 600.0, "f_max": 550.0},
        {"yin_threshold": 1.0},
        {"periodicity_threshold": 1.1},
        {"frame_ms": 10.0, "f_min": 50.0},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        PitchConfig(**kwargs)
