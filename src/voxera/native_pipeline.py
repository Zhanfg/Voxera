from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .acoustic import MelSpectrogram
from .audio import AudioBuffer, resample_linear
from .conditioning import FrameConditions, align_frame_conditions
from .features import FeatureBatch, LogMelFrontend
from .pitch import PitchTrack, YinPitchExtractor
from .speaker import SpeakerEmbedding
from .vocoder import SpectralFrames, StreamingISTFT

FloatArray = NDArray[np.float32]


class ContentModel(Protocol):
    def encode(self, features: FloatArray) -> FloatArray:
        """Return dense [frames, 256] content features at 10 ms cadence."""
        ...


class SpeakerModel(Protocol):
    def encode(self, features: FloatArray) -> SpeakerEmbedding:
        """Return one normalized target-speaker embedding."""
        ...


class ConditionModel(Protocol):
    def fuse(self, conditions: FrameConditions) -> FloatArray:
        """Return [frames, 256] decoder conditions at 40 ms cadence."""
        ...


class AcousticDecoderModel(Protocol):
    def decode(self, conditions: FloatArray) -> MelSpectrogram:
        """Return 80-bin log-mel frames at 10 ms cadence."""
        ...


class VocoderModel(Protocol):
    def predict(self, mel: MelSpectrogram) -> SpectralFrames:
        """Return one Fourier synthesis frame per mel frame."""
        ...


@dataclass(frozen=True, slots=True)
class NativeModels:
    content: ContentModel
    speaker: SpeakerModel
    condition: ConditionModel
    decoder: AcousticDecoderModel
    vocoder: VocoderModel


@dataclass(frozen=True, slots=True)
class NativeConversionTrace:
    """Compact shape/time trace for debugging an end-to-end conversion."""

    source_samples: int
    source_feature_frames: int
    pitch_frames: int
    condition_frames: int
    mel_frames: int
    spectral_frames: int
    output_samples: int


@dataclass(frozen=True, slots=True)
class NativeConversionResult:
    audio: AudioBuffer
    trace: NativeConversionTrace


class NativeOfflinePipeline:
    """Compose Voxera-native components into an inspectable WAV-to-WAV path.

    Neural execution is supplied through small protocols so the orchestration is
    independent from PyTorch, ONNX Runtime, QNN, or another edge backend.
    """

    def __init__(
        self,
        models: NativeModels,
        *,
        sample_rate: int = 16_000,
    ) -> None:
        if sample_rate != 16_000:
            raise ValueError("M1 native pipeline currently requires 16 kHz")
        self.models = models
        self.sample_rate = sample_rate
        self.frontend = LogMelFrontend()
        self.pitch_extractor = YinPitchExtractor()

    def prepare_speaker(self, reference_audio: AudioBuffer) -> SpeakerEmbedding:
        reference = self._prepare_audio(reference_audio)
        features = self.frontend.extract(reference.samples)
        if features.frame_count == 0:
            raise ValueError("reference audio is too short to extract speaker features")
        speaker = self.models.speaker.encode(features.values)
        if speaker.dimension != 256:
            raise ValueError(
                f"speaker model must return 256 dimensions, got {speaker.dimension}"
            )
        return speaker

    def convert(
        self,
        source_audio: AudioBuffer,
        speaker: SpeakerEmbedding,
    ) -> NativeConversionResult:
        if speaker.dimension != 256:
            raise ValueError("native M1 pipeline expects a 256-d speaker embedding")

        source = self._prepare_audio(source_audio)
        feature_batch = self.frontend.extract(source.samples)
        pitch_track = self.pitch_extractor.extract(source.samples)

        dense_content = self._validate_dense_content(
            self.models.content.encode(feature_batch.values),
            feature_batch,
        )
        frame_conditions = align_frame_conditions(
            dense_content,
            pitch_track,
            speaker,
        )

        fused = self._validate_fused(
            self.models.condition.fuse(frame_conditions),
            frame_conditions,
        )
        mel = self.models.decoder.decode(fused)
        self._validate_mel(mel, frame_conditions)

        spectral = self.models.vocoder.predict(mel)
        self._validate_spectral(spectral, mel)

        synthesizer = StreamingISTFT()
        pcm = synthesizer.push(spectral)
        output = AudioBuffer(pcm, self.sample_rate)

        return NativeConversionResult(
            audio=output,
            trace=NativeConversionTrace(
                source_samples=source.samples.size,
                source_feature_frames=feature_batch.frame_count,
                pitch_frames=pitch_track.frame_count,
                condition_frames=frame_conditions.frame_count,
                mel_frames=mel.frame_count,
                spectral_frames=spectral.frame_count,
                output_samples=pcm.size,
            ),
        )

    def convert_with_reference(
        self,
        source_audio: AudioBuffer,
        reference_audio: AudioBuffer,
    ) -> NativeConversionResult:
        speaker = self.prepare_speaker(reference_audio)
        return self.convert(source_audio, speaker)

    def _prepare_audio(self, audio: AudioBuffer) -> AudioBuffer:
        return resample_linear(audio, self.sample_rate)

    @staticmethod
    def _validate_dense_content(
        values: FloatArray,
        features: FeatureBatch,
    ) -> FloatArray:
        content = np.asarray(values, dtype=np.float32)
        if content.ndim != 2 or content.shape[1] != 256:
            raise ValueError("content model must return [frames, 256]")
        if content.shape[0] != features.frame_count:
            raise ValueError(
                "content model must preserve the dense acoustic frame count"
            )
        if not np.all(np.isfinite(content)):
            raise ValueError("content model returned non-finite values")
        return content

    @staticmethod
    def _validate_fused(
        values: FloatArray,
        conditions: FrameConditions,
    ) -> FloatArray:
        fused = np.asarray(values, dtype=np.float32)
        expected = (conditions.frame_count, 256)
        if fused.shape != expected:
            raise ValueError(
                f"condition model must return shape {expected}, got {fused.shape}"
            )
        if not np.all(np.isfinite(fused)):
            raise ValueError("condition model returned non-finite values")
        return fused

    @staticmethod
    def _validate_mel(
        mel: MelSpectrogram,
        conditions: FrameConditions,
    ) -> None:
        expected_frames = conditions.frame_count * 4
        if mel.frame_count != expected_frames:
            raise ValueError(
                f"decoder must emit {expected_frames} mel frames, "
                f"got {mel.frame_count}"
            )

    @staticmethod
    def _validate_spectral(
        spectral: SpectralFrames,
        mel: MelSpectrogram,
    ) -> None:
        if spectral.frame_count != mel.frame_count:
            raise ValueError(
                "vocoder must emit one spectral frame per mel frame"
            )
        if spectral.frequency_bins != 161:
            raise ValueError(
                f"vocoder must emit 161 frequency bins, "
                f"got {spectral.frequency_bins}"
            )
