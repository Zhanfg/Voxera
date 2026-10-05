"""Voxera reference runtime."""

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
    "LogMelFrontend",
    "PitchConfig",
    "PitchTrack",
    "ReferencePipeline",
    "SpeakerEmbedding",
    "StreamingCadenceSelector",
    "StreamingLogMelFrontend",
    "StreamingYinPitchExtractor",
    "YinPitchExtractor",
    "cosine_similarity",
]
__version__ = "0.1.0a4"
