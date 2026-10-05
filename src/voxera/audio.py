from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatAudio = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    samples: FloatAudio
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("AudioBuffer expects mono audio")
        if not np.all(np.isfinite(samples)):
            raise ValueError("audio contains non-finite values")
        object.__setattr__(self, "samples", samples)

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / self.sample_rate


def load_wav(path: str | Path) -> AudioBuffer:
    """Read uncompressed 16-bit PCM WAV and return mono float32 audio."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError("reference WAV reader currently supports 16-bit PCM only")
    if channels <= 0:
        raise ValueError("WAV must contain at least one channel")

    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        if pcm.size % channels:
            raise ValueError("corrupt interleaved WAV data")
        pcm = pcm.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return AudioBuffer(pcm.astype(np.float32, copy=False), sample_rate)


def write_wav(path: str | Path, audio: AudioBuffer) -> None:
    """Write mono audio as 16-bit PCM WAV."""

    clipped = np.clip(audio.samples, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(pcm.tobytes())


def resample_linear(audio: AudioBuffer, target_rate: int) -> AudioBuffer:
    """Dependency-light deterministic resampler for the reference path.

    This is intentionally not the production resampler. Platform runtimes can
    replace it without changing the model-facing audio contract.
    """

    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if target_rate == audio.sample_rate or audio.samples.size == 0:
        return AudioBuffer(audio.samples.copy(), target_rate)

    new_length = max(1, round(audio.samples.size * target_rate / audio.sample_rate))
    old_positions = np.arange(audio.samples.size, dtype=np.float64)
    new_positions = np.arange(new_length, dtype=np.float64) * audio.sample_rate / target_rate
    new_positions = np.minimum(new_positions, audio.samples.size - 1)
    samples = np.interp(new_positions, old_positions, audio.samples).astype(np.float32)
    return AudioBuffer(samples, target_rate)


def normalize_peak(audio: AudioBuffer, peak: float = 0.98) -> AudioBuffer:
    if not 0.0 < peak <= 1.0:
        raise ValueError("peak must be in (0, 1]")
    magnitude = float(np.max(np.abs(audio.samples), initial=0.0))
    if magnitude == 0.0 or magnitude <= peak:
        return AudioBuffer(audio.samples.copy(), audio.sample_rate)
    return AudioBuffer((audio.samples * (peak / magnitude)).astype(np.float32), audio.sample_rate)
