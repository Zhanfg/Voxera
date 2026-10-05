"""Voxera reference runtime."""

from .acoustic import MelConfig, MelSpectrogram, expected_mel_frames
from .conditioning import FrameConditions, align_frame_conditions, pitch_condition_features
from .config import AudioConfig
from .content import ContentCadence, StreamingCadenceSelector
from .features import FbankConfig, FeatureBatch, LogMelFrontend, StreamingLogMelFrontend
from .native_pipeline import (
    NativeConversionResult,
    NativeConversionTrace,
    NativeModels,
    NativeOfflinePipeline,
)
from .pipeline import ReferencePipeline
from .pitch import PitchConfig, PitchTrack, StreamingYinPitchExtractor, YinPitchExtractor
from .prosody import (
    PROSODY_GLOBAL_DIM,
    PROSODY_GLOBAL_EMBED_DIM,
    PROSODY_LOCAL_DIM,
    PROSODY_LOCAL_EMBED_DIM,
    PROSODY_STYLES,
    NativeProsodyExtractor,
    ProsodyEmbedding,
    ProsodyTrack,
    classify_prosody_style,
)
from .speaker import SpeakerEmbedding, cosine_similarity
from .vocoder import (
    SpectralFrames,
    SpectralSynthesisConfig,
    StreamingISTFT,
    waveform_to_spectral_frames,
)

__all__ = [
    "AudioConfig",
    "ContentCadence",
    "FbankConfig",
    "FeatureBatch",
    "FrameConditions",
    "LogMelFrontend",
    "MelConfig",
    "MelSpectrogram",
    "NativeConversionResult",
    "NativeConversionTrace",
    "NativeModels",
    "NativeOfflinePipeline",
    "PitchConfig",
    "PitchTrack",
    "PROSODY_GLOBAL_DIM",
    "PROSODY_GLOBAL_EMBED_DIM",
    "PROSODY_LOCAL_DIM",
    "PROSODY_LOCAL_EMBED_DIM",
    "PROSODY_STYLES",
    "ProsodyEmbedding",
    "ProsodyTrack",
    "ReferencePipeline",
    "NativeProsodyExtractor",
    "SpeakerEmbedding",
    "SpectralFrames",
    "SpectralSynthesisConfig",
    "StreamingCadenceSelector",
    "StreamingISTFT",
    "StreamingLogMelFrontend",
    "StreamingYinPitchExtractor",
    "YinPitchExtractor",
    "align_frame_conditions",
    "classify_prosody_style",
    "cosine_similarity",
    "expected_mel_frames",
    "pitch_condition_features",
    "waveform_to_spectral_frames",
]
__version__ = "0.2.0a1"
