from __future__ import annotations

import numpy as np
import pytest

from voxera.vocoder import (
    SpectralFrames,
    SpectralSynthesisConfig,
    StreamingISTFT,
    waveform_to_spectral_frames,
)


def _test_waveform(seconds: float = 0.2) -> np.ndarray:
    sample_rate = 16_000
    time = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (
        0.1 * np.sin(2.0 * np.pi * 110.0 * time)
        + 0.04 * np.sin(2.0 * np.pi * 730.0 * time)
    ).astype(np.float32)


def test_spectral_targets_reconstruct_waveform() -> None:
    waveform = _test_waveform()
    frames = waveform_to_spectral_frames(waveform)

    reconstructed = StreamingISTFT().push(frames)

    assert reconstructed.shape == waveform.shape
    assert np.max(np.abs(reconstructed - waveform)) < 2e-5


def test_chunked_synthesis_matches_single_push() -> None:
    waveform = _test_waveform(0.4)
    frames = waveform_to_spectral_frames(waveform)

    offline = StreamingISTFT().push(frames)
    streaming = StreamingISTFT()
    outputs = []

    cursor = 0
    for count in (1, 3, 2, 7, 4):
        end = min(frames.frame_count, cursor + count)
        chunk = SpectralFrames(
            frames.log_magnitude[cursor:end],
            frames.phase[cursor:end],
        )
        outputs.append(streaming.push(chunk))
        cursor = end
        if cursor == frames.frame_count:
            break

    if cursor < frames.frame_count:
        outputs.append(
            streaming.push(
                SpectralFrames(
                    frames.log_magnitude[cursor:],
                    frames.phase[cursor:],
                )
            )
        )

    actual = np.concatenate(outputs)
    assert np.array_equal(actual, offline)


def test_empty_spectral_batch_emits_no_samples() -> None:
    cfg = SpectralSynthesisConfig()
    empty = np.empty((0, cfg.frequency_bins), dtype=np.float32)
    output = StreamingISTFT(cfg).push(SpectralFrames(empty, empty.copy()))
    assert output.size == 0


def test_wrong_bin_count_is_rejected() -> None:
    cfg = SpectralSynthesisConfig()
    frames = SpectralFrames(
        np.zeros((2, cfg.frequency_bins - 1), dtype=np.float32),
        np.zeros((2, cfg.frequency_bins - 1), dtype=np.float32),
    )
    with pytest.raises(ValueError):
        StreamingISTFT(cfg).push(frames)


def test_non_hop_aligned_waveform_is_rejected() -> None:
    with pytest.raises(ValueError):
        waveform_to_spectral_frames(np.zeros(161, dtype=np.float32))
