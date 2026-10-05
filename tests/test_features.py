from __future__ import annotations

import numpy as np
import pytest

from voxera.features import FbankConfig, LogMelFrontend, StreamingLogMelFrontend


def test_default_frontend_shape() -> None:
    frontend = LogMelFrontend()
    samples = np.zeros(16_000, dtype=np.float32)

    features = frontend.extract(samples)

    assert features.values.shape == (98, 80)
    assert features.sample_rate == 16_000
    assert features.frame_samples == 400
    assert features.hop_samples == 160
    assert np.all(np.isfinite(features.values))


def test_streaming_matches_offline_for_irregular_chunks() -> None:
    rng = np.random.default_rng(7)
    samples = rng.normal(0.0, 0.1, 32_000).astype(np.float32)

    offline = LogMelFrontend().extract(samples).values
    streaming = StreamingLogMelFrontend()

    chunks = [1, 17, 80, 511, 2_048, 77, 4_096, 333, 8_000]
    outputs = []
    cursor = 0
    for size in chunks:
        end = min(samples.size, cursor + size)
        batch = streaming.push(samples[cursor:end])
        if batch.frame_count:
            outputs.append(batch.values)
        cursor = end
        if cursor == samples.size:
            break

    if cursor < samples.size:
        outputs.append(streaming.push(samples[cursor:]).values)

    combined = np.concatenate(outputs, axis=0)
    assert combined.shape == offline.shape
    assert np.allclose(combined, offline, atol=2e-6)


def test_short_audio_produces_no_frames() -> None:
    config = FbankConfig()
    frontend = LogMelFrontend(config)

    features = frontend.extract(np.zeros(config.frame_samples - 1, dtype=np.float32))

    assert features.frame_count == 0
    assert features.feature_dim == config.n_mels


def test_sine_energy_is_localized_in_frequency() -> None:
    config = FbankConfig(preemphasis=0.0)
    frontend = LogMelFrontend(config)
    time = np.arange(config.sample_rate, dtype=np.float32) / config.sample_rate
    samples = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)

    features = frontend.extract(samples).values
    mean_log_energy = features.mean(axis=0)

    assert float(mean_log_energy.max() - np.median(mean_log_energy)) > 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"n_mels": 0},
        {"n_fft": 128},
        {"preemphasis": 1.0},
        {"f_min": 8_000.0},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        FbankConfig(**kwargs)
