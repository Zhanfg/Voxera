from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .content import ContentCadence
from .pitch import PitchTrack
from .speaker import SpeakerEmbedding

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FrameConditions:
    """Aligned per-frame conditions consumed by the acoustic decoder."""

    content: FloatArray
    pitch: FloatArray
    speaker: SpeakerEmbedding
    frame_period_seconds: float

    def __post_init__(self) -> None:
        content = np.asarray(self.content, dtype=np.float32)
        pitch = np.asarray(self.pitch, dtype=np.float32)

        if content.ndim != 2:
            raise ValueError("content must be a 2-D time-major array")
        if pitch.ndim != 2 or pitch.shape[1] != 3:
            raise ValueError("pitch must have shape [frames, 3]")
        if content.shape[0] != pitch.shape[0]:
            raise ValueError("content and pitch must have the same frame count")
        if not np.all(np.isfinite(content)) or not np.all(np.isfinite(pitch)):
            raise ValueError("frame conditions contain non-finite values")
        if self.frame_period_seconds <= 0.0:
            raise ValueError("frame_period_seconds must be positive")

        object.__setattr__(self, "content", content)
        object.__setattr__(self, "pitch", pitch)

    @property
    def frame_count(self) -> int:
        return self.content.shape[0]

    @property
    def content_dim(self) -> int:
        return self.content.shape[1]


def pitch_condition_features(
    track: PitchTrack,
    *,
    reference_hz: float = 55.0,
) -> FloatArray:
    """Encode F0, voicing, and periodicity into a small stable feature vector.

    The first channel is log2(F0/reference_hz) for voiced frames and zero for
    unvoiced frames. The remaining channels are voiced probability (0/1) and
    periodicity.
    """

    if reference_hz <= 0.0:
        raise ValueError("reference_hz must be positive")

    log_f0 = np.zeros(track.frame_count, dtype=np.float32)
    voiced = track.voiced.astype(np.float32)
    valid = track.voiced & (track.f0_hz > 0.0)
    if np.any(valid):
        log_f0[valid] = np.log2(track.f0_hz[valid] / reference_hz).astype(
            np.float32
        )

    return np.stack(
        (
            log_f0,
            voiced,
            track.periodicity.astype(np.float32, copy=False),
        ),
        axis=1,
    )


def align_frame_conditions(
    dense_content: FloatArray,
    pitch_track: PitchTrack,
    speaker: SpeakerEmbedding,
    cadence: ContentCadence | None = None,
) -> FrameConditions:
    """Align dense 10 ms content and pitch streams onto the VC cadence.

    Content and F0 use different analysis-window lengths, so their tail frame
    counts can differ even with the same hop size. Voxera aligns by frame start
    time and crops to the common prefix before applying the shared cadence.
    """

    content = np.asarray(dense_content, dtype=np.float32)
    if content.ndim != 2:
        raise ValueError("dense_content must be a 2-D time-major array")
    if not np.all(np.isfinite(content)):
        raise ValueError("dense_content contains non-finite values")
    if pitch_track.hop_samples <= 0 or pitch_track.sample_rate <= 0:
        raise ValueError("pitch track has an invalid timebase")

    cadence = cadence or ContentCadence()
    pitch = pitch_condition_features(pitch_track)

    common_frames = min(content.shape[0], pitch.shape[0])
    content = content[:common_frames]
    pitch = pitch[:common_frames]

    indices = np.arange(
        cadence.offset,
        common_frames,
        cadence.stride,
        dtype=np.int64,
    )
    aligned_content = content[indices].copy()
    aligned_pitch = pitch[indices].copy()

    frame_period_seconds = (
        pitch_track.hop_samples * cadence.stride / pitch_track.sample_rate
    )
    return FrameConditions(
        content=aligned_content,
        pitch=aligned_pitch,
        speaker=speaker,
        frame_period_seconds=frame_period_seconds,
    )
