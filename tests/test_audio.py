import numpy as np

from voxera.audio import AudioBuffer, load_wav, normalize_peak, resample_linear, write_wav


def test_resample_preserves_duration() -> None:
    samples = np.linspace(-0.5, 0.5, 16_000, dtype=np.float32)
    source = AudioBuffer(samples, 16_000)
    target = resample_linear(source, 24_000)
    assert target.sample_rate == 24_000
    assert target.samples.size == 24_000
    assert abs(target.duration_seconds - source.duration_seconds) < 1e-6


def test_normalize_peak_only_attenuates_when_needed() -> None:
    audio = AudioBuffer(np.array([-1.0, 0.5], dtype=np.float32), 16_000)
    normalized = normalize_peak(audio, 0.8)
    assert np.isclose(np.max(np.abs(normalized.samples)), 0.8)


def test_wav_roundtrip(tmp_path) -> None:
    source = AudioBuffer(np.array([-0.5, 0.0, 0.5], dtype=np.float32), 16_000)
    path = tmp_path / "sample.wav"
    write_wav(path, source)
    restored = load_wav(path)
    assert restored.sample_rate == 16_000
    assert np.allclose(restored.samples, source.samples, atol=1.0 / 32768.0)
