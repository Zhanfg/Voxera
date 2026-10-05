from __future__ import annotations

import numpy as np

from voxera.pitch import PitchTrack
from voxera.prosody import NativeProsodyExtractor, PROSODY_GLOBAL_DIM, PROSODY_LOCAL_DIM


def _track(
    f0: np.ndarray,
    *,
    periodicity: float = 0.9,
) -> PitchTrack:
    voiced = f0 > 0.0
    return PitchTrack(
        f0_hz=f0.astype(np.float32),
        periodicity=np.where(voiced, periodicity, 0.0).astype(np.float32),
        voiced=voiced,
        sample_rate=16_000,
        frame_samples=640,
        hop_samples=160,
    )


def test_prosody_descriptor_shapes_and_finiteness() -> None:
    samples = np.zeros(8_000, dtype=np.float32)
    f0 = np.full(47, 180.0, dtype=np.float32)

    result = NativeProsodyExtractor().extract(samples, _track(f0))

    assert result.local.shape[1] == PROSODY_LOCAL_DIM
    assert result.global_style.shape == (PROSODY_GLOBAL_DIM,)
    assert result.frame_count == 11
    assert np.all(np.isfinite(result.local))
    assert np.all(np.isfinite(result.global_style))
    assert 0.0 <= result.confidence <= 1.0


def test_rising_final_contour_is_detected() -> None:
    frames = 51
    time = np.arange(frames, dtype=np.float32)
    f0 = 150.0 + np.maximum(time - 38.0, 0.0) * 8.0
    samples = 0.08 * np.sin(
        2.0 * np.pi * 180.0 * np.arange(8_640, dtype=np.float32) / 16_000.0
    )

    result = NativeProsodyExtractor().extract(samples, _track(f0))

    assert result.global_style[6] > 0.0
    assert result.style in {"rising", "animated"}


def test_silence_is_subdued_and_pause_heavy() -> None:
    samples = np.zeros(8_000, dtype=np.float32)
    f0 = np.zeros(47, dtype=np.float32)

    result = NativeProsodyExtractor().extract(samples, _track(f0, periodicity=0.0))

    assert result.style == "subdued"
    assert result.global_style[5] >= 0.9
    assert result.global_style[4] == 0.0


def test_emphasis_proxy_reacts_to_energy_burst() -> None:
    samples = np.full(8_000, 0.01, dtype=np.float32)
    samples[3_200:4_800] = 0.35
    f0 = np.full(47, 160.0, dtype=np.float32)

    result = NativeProsodyExtractor().extract(samples, _track(f0))

    assert float(np.max(result.local[:, 7])) > 0.62
