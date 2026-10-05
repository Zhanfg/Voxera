from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .content import ContentCadence
from .pitch import PitchTrack

FloatArray = NDArray[np.float32]


PROSODY_LOCAL_DIM = 8
PROSODY_GLOBAL_DIM = 8
PROSODY_LOCAL_EMBED_DIM = 32
PROSODY_GLOBAL_EMBED_DIM = 16
PROSODY_STYLES = ("neutral", "rising", "emphatic", "animated", "subdued")


@dataclass(frozen=True, slots=True)
class ProsodyEmbedding:
    """Learned local + phrase-level prosody representation for M2 conditioning."""

    local: FloatArray
    global_style: FloatArray

    def __post_init__(self) -> None:
        local = np.asarray(self.local, dtype=np.float32)
        global_style = np.asarray(self.global_style, dtype=np.float32)
        if local.ndim != 2 or local.shape[1] != PROSODY_LOCAL_EMBED_DIM:
            raise ValueError(
                f"local embedding must have shape [frames, {PROSODY_LOCAL_EMBED_DIM}]"
            )
        if global_style.shape != (PROSODY_GLOBAL_EMBED_DIM,):
            raise ValueError(
                "global style embedding must have shape "
                f"[{PROSODY_GLOBAL_EMBED_DIM}]"
            )
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(global_style)):
            raise ValueError("prosody embedding contains non-finite values")
        object.__setattr__(self, "local", local)
        object.__setattr__(self, "global_style", global_style)

    @property
    def frame_count(self) -> int:
        return self.local.shape[0]


@dataclass(frozen=True, slots=True)
class ProsodyTrack:
    """Interpretable acoustic delivery features on Voxera's 40 ms cadence.

    Local channels:
      0 robust-normalized log-F0
      1 local log-F0 delta
      2 periodicity
      3 voiced flag
      4 robust-normalized log-energy
      5 local log-energy delta
      6 pause likelihood
      7 emphasis likelihood

    Global channels:
      0 median pitch level relative to 110 Hz
      1 voiced pitch range (10th-90th percentile, octaves)
      2 mean loudness proxy
      3 energy dynamics
      4 voiced ratio
      5 pause ratio
      6 final pitch contour
      7 emphasis ratio
    """

    local: FloatArray
    global_style: FloatArray
    style: str
    confidence: float
    frame_period_seconds: float

    def __post_init__(self) -> None:
        local = np.asarray(self.local, dtype=np.float32)
        global_style = np.asarray(self.global_style, dtype=np.float32)

        if local.ndim != 2 or local.shape[1] != PROSODY_LOCAL_DIM:
            raise ValueError(
                f"local prosody must have shape [frames, {PROSODY_LOCAL_DIM}]"
            )
        if global_style.shape != (PROSODY_GLOBAL_DIM,):
            raise ValueError(
                f"global prosody must have shape [{PROSODY_GLOBAL_DIM}]"
            )
        if not np.all(np.isfinite(local)) or not np.all(np.isfinite(global_style)):
            raise ValueError("prosody contains non-finite values")
        if self.style not in PROSODY_STYLES:
            raise ValueError(f"unsupported prosody style: {self.style}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.frame_period_seconds <= 0.0:
            raise ValueError("frame_period_seconds must be positive")

        object.__setattr__(self, "local", local)
        object.__setattr__(self, "global_style", global_style)

    @property
    def frame_count(self) -> int:
        return self.local.shape[0]


class NativeProsodyExtractor:
    """Dependency-light sentence-delivery descriptor.

    This is not an emotion classifier. It extracts measurable acoustic style so
    Voxera can represent emphasis, pauses, pitch contour, dynamics, and delivery
    intensity before a learned ProsodyNet is available.
    """

    def __init__(self, cadence: ContentCadence | None = None) -> None:
        self.cadence = cadence or ContentCadence()

    def extract(self, samples: FloatArray, pitch: PitchTrack) -> ProsodyTrack:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("prosody extractor expects mono 1-D audio")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains non-finite values")
        if pitch.sample_rate <= 0 or pitch.hop_samples <= 0 or pitch.frame_samples <= 0:
            raise ValueError("pitch track has an invalid timebase")

        frame_energy = _frame_log_energy(
            audio,
            pitch.frame_samples,
            pitch.hop_samples,
            pitch.frame_count,
        )
        common = min(pitch.frame_count, frame_energy.size)
        if common == 0:
            return ProsodyTrack(
                local=np.empty((0, PROSODY_LOCAL_DIM), dtype=np.float32),
                global_style=np.zeros(PROSODY_GLOBAL_DIM, dtype=np.float32),
                style="subdued",
                confidence=1.0,
                frame_period_seconds=(
                    pitch.hop_samples * self.cadence.stride / pitch.sample_rate
                ),
            )

        f0 = pitch.f0_hz[:common]
        periodicity = pitch.periodicity[:common]
        voiced = pitch.voiced[:common]
        energy = frame_energy[:common]

        log_f0 = np.zeros(common, dtype=np.float32)
        valid = voiced & (f0 > 0.0)
        if np.any(valid):
            log_f0[valid] = np.log2(f0[valid]).astype(np.float32)

        pitch_z = _robust_voiced_z(log_f0, valid)
        energy_z = _robust_z(energy)

        pitch_delta = _delta(pitch_z)
        energy_delta = _delta(energy_z)

        energy_floor = float(np.percentile(energy, 30))
        pause = np.clip(
            (energy_floor + 0.35 - energy) / 1.25,
            0.0,
            1.0,
        ).astype(np.float32)
        pause = np.maximum(pause, (~voiced).astype(np.float32) * 0.90)

        emphasis_drive = (
            0.55 * np.maximum(energy_z, 0.0)
            + 0.30 * np.maximum(pitch_z, 0.0)
            + 0.15 * np.abs(pitch_delta)
        )
        emphasis = _soft_unit(emphasis_drive - 0.55)

        dense = np.stack(
            (
                pitch_z,
                pitch_delta,
                periodicity.astype(np.float32, copy=False),
                voiced.astype(np.float32),
                energy_z,
                energy_delta,
                pause,
                emphasis,
            ),
            axis=1,
        ).astype(np.float32)

        indices = np.arange(
            self.cadence.offset,
            common,
            self.cadence.stride,
            dtype=np.int64,
        )
        local = dense[indices].copy()

        global_style = _global_descriptor(
            f0=f0,
            voiced=voiced,
            energy=energy,
            local=local,
        )
        style, confidence = classify_prosody_style(global_style)

        return ProsodyTrack(
            local=local,
            global_style=global_style,
            style=style,
            confidence=confidence,
            frame_period_seconds=(
                pitch.hop_samples * self.cadence.stride / pitch.sample_rate
            ),
        )


def _frame_log_energy(
    audio: FloatArray,
    frame_samples: int,
    hop_samples: int,
    frame_count: int,
) -> FloatArray:
    if frame_count <= 0:
        return np.empty(0, dtype=np.float32)

    energies = np.empty(frame_count, dtype=np.float32)
    eps = np.float32(1e-7)
    for index in range(frame_count):
        start = index * hop_samples
        frame = audio[start : start + frame_samples]
        if frame.size < frame_samples:
            padded = np.zeros(frame_samples, dtype=np.float32)
            padded[: frame.size] = frame
            frame = padded
        rms = np.sqrt(np.mean(frame * frame, dtype=np.float64)).astype(np.float32)
        energies[index] = np.log(np.maximum(rms, eps))
    return energies


def _robust_voiced_z(values: FloatArray, valid: NDArray[np.bool_]) -> FloatArray:
    result = np.zeros_like(values, dtype=np.float32)
    if not np.any(valid):
        return result
    center = float(np.median(values[valid]))
    scale = _robust_scale(values[valid])
    result[valid] = np.clip((values[valid] - center) / scale, -4.0, 4.0)
    return result


def _robust_z(values: FloatArray) -> FloatArray:
    center = float(np.median(values))
    scale = _robust_scale(values)
    return np.clip((values - center) / scale, -4.0, 4.0).astype(np.float32)


def _robust_scale(values: FloatArray) -> float:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale < 1e-4:
        scale = float(np.std(values))
    return max(scale, 1e-4)


def _delta(values: FloatArray) -> FloatArray:
    result = np.zeros_like(values, dtype=np.float32)
    if values.size > 1:
        result[1:] = values[1:] - values[:-1]
    return np.clip(result, -4.0, 4.0)


def _soft_unit(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, -8.0, 8.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _global_descriptor(
    *,
    f0: FloatArray,
    voiced: NDArray[np.bool_],
    energy: FloatArray,
    local: FloatArray,
) -> FloatArray:
    valid_f0 = f0[voiced & (f0 > 0.0)]
    if valid_f0.size:
        median_f0 = float(np.median(valid_f0))
        pitch_level = float(np.clip(np.log2(median_f0 / 110.0), -2.0, 2.0))
        log_voiced = np.log2(valid_f0)
        pitch_range = float(
            np.clip(
                np.percentile(log_voiced, 90) - np.percentile(log_voiced, 10),
                0.0,
                2.0,
            )
        )
    else:
        pitch_level = 0.0
        pitch_range = 0.0

    mean_dbfs = float(np.mean(energy) * (20.0 / np.log(10.0)))
    loudness = float(np.clip((mean_dbfs + 60.0) / 60.0, 0.0, 1.0))
    energy_dynamics = float(np.clip(np.std(energy) / 2.0, 0.0, 1.0))
    voiced_ratio = float(np.mean(voiced.astype(np.float32)))

    if local.shape[0]:
        pause_ratio = float(np.mean(local[:, 6]))
        emphasis_ratio = float(np.mean(local[:, 7] > 0.62))
        voiced_local = local[:, 3] > 0.5
        pitch_local = local[:, 0]
        voiced_indices = np.flatnonzero(voiced_local)
        if voiced_indices.size >= 2:
            tail = voiced_indices[-min(6, voiced_indices.size) :]
            final_contour = float(
                np.clip(
                    (pitch_local[tail[-1]] - pitch_local[tail[0]])
                    / max(1, tail.size - 1),
                    -1.0,
                    1.0,
                )
            )
        else:
            final_contour = 0.0
    else:
        pause_ratio = 1.0
        emphasis_ratio = 0.0
        final_contour = 0.0

    return np.asarray(
        (
            pitch_level,
            pitch_range,
            loudness,
            energy_dynamics,
            voiced_ratio,
            pause_ratio,
            final_contour,
            emphasis_ratio,
        ),
        dtype=np.float32,
    )


def classify_prosody_style(global_style: FloatArray) -> tuple[str, float]:
    """Map an 8-D acoustic style descriptor to a coarse delivery label."""

    descriptor = np.asarray(global_style, dtype=np.float32)
    if descriptor.shape != (PROSODY_GLOBAL_DIM,):
        raise ValueError(
            f"global_style must have shape [{PROSODY_GLOBAL_DIM}]"
        )
    if not np.all(np.isfinite(descriptor)):
        raise ValueError("global_style contains non-finite values")

    global_style = descriptor

    (
        _pitch_level,
        pitch_range,
        loudness,
        dynamics,
        voiced_ratio,
        pause_ratio,
        final_contour,
        emphasis_ratio,
    ) = (float(value) for value in global_style)

    scores = {
        "neutral": 0.45,
        "rising": max(0.0, final_contour) * 1.6 + pitch_range * 0.15,
        "emphatic": emphasis_ratio * 1.3 + dynamics * 0.55 + loudness * 0.25,
        "animated": pitch_range * 0.65 + dynamics * 0.75 + loudness * 0.20,
        "subdued": (
            (1.0 - loudness) * 0.65
            + pause_ratio * 0.45
            + (1.0 - voiced_ratio) * 0.20
            + max(0.0, 0.35 - pitch_range)
        ),
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    style, best = ranked[0]
    second = ranked[1][1]
    margin = max(0.0, best - second)
    confidence = float(np.clip(0.5 + margin * 0.45, 0.5, 0.98))
    return style, confidence
