"""Voxera reference runtime."""

from .conditioning import FrameConditions, align_frame_conditions, pitch_condition_features
from .config import AudioConfig
from .content import ContentCadence, StreamingCadenceSelector
from .features import FbankConfig, FeatureBatch, LogMelFrontend, StreamingLogMelFrontend
from .pipeline import ReferencePipeline
from .pitch import PitchConfig, PitchTrack, StreamingYinPitchExtractor, YinPitchExtractor
from .speaker import SpeakerEmbedding, cosine_similarity

__all__ = [
    "AudioConfig",
    "ContentCadence",
    "FbankConfig",
    "FeatureBatch",
    "FrameConditions",
    "LogMelFrontend",
    "PitchConfig",
    "PitchTrack",
    "ReferencePipeline",
    "SpeakerEmbedding",
    "StreamingCadenceSelector",
    "StreamingLogMelFrontend",
    "StreamingYinPitchExtractor",
    "YinPitchExtractor",
    "align_frame_conditions",
    "cosine_similarity",
    "pitch_condition_features",
]
__version__ = "0.1.0a5"
