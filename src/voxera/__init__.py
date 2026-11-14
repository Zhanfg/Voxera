"""Voxera reference runtime."""

from .acoustic import MelConfig, MelSpectrogram, expected_mel_frames
from .asr import SemanticMailbox, SemanticSidecar, StreamingASRBackend, TextReplayASR
from .asr_backends import ASRBackendKind, ASRBackendSelection, create_asr_backend
from .asr_sherpa import SherpaOnnxStreamingASR, SherpaOnnxTransducerConfig
from .asr_whisper import WhisperCppCliASR, WhisperCppCliConfig
from .conditioning import FrameConditions, align_frame_conditions, pitch_condition_features
from .config import AudioConfig
from .content import ContentCadence, StreamingCadenceSelector
from .deployment import (
    DEPLOYMENT_SCHEMA,
    BundleManifest,
    ModelArtifact,
    PrecisionPolicy,
    TensorSpec,
    load_bundle_manifest,
    verify_bundle,
)
from .features import FbankConfig, FeatureBatch, LogMelFrontend, StreamingLogMelFrontend
from .hypothesis import HypothesisStabilizer, HypothesisStabilizerConfig
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
from .semantic import (
    SEMANTIC_FEATURE_DIM,
    SEMANTIC_MODES,
    NativeSemanticAnalyzer,
    SemanticSnapshot,
    TranscriptHypothesis,
)
from .speaker import SpeakerEmbedding, cosine_similarity
from .vocoder import (
    SpectralFrames,
    SpectralSynthesisConfig,
    StreamingISTFT,
    waveform_to_spectral_frames,
)

__all__ = [
    "ASRBackendKind",
    "ASRBackendSelection",
    "AudioConfig",
    "BundleManifest",
    "DEPLOYMENT_SCHEMA",
    "NativeSemanticAnalyzer",
    "ContentCadence",
    "FbankConfig",
    "FeatureBatch",
    "FrameConditions",
    "HypothesisStabilizer",
    "HypothesisStabilizerConfig",
    "LogMelFrontend",
    "MelConfig",
    "MelSpectrogram",
    "ModelArtifact",
    "NativeConversionResult",
    "NativeConversionTrace",
    "NativeModels",
    "NativeOfflinePipeline",
    "PitchConfig",
    "PrecisionPolicy",
    "PitchTrack",
    "PROSODY_GLOBAL_DIM",
    "PROSODY_GLOBAL_EMBED_DIM",
    "PROSODY_LOCAL_DIM",
    "PROSODY_LOCAL_EMBED_DIM",
    "PROSODY_STYLES",
    "ProsodyEmbedding",
    "ProsodyTrack",
    "ReferencePipeline",
    "SEMANTIC_FEATURE_DIM",
    "SEMANTIC_MODES",
    "SemanticMailbox",
    "SemanticSidecar",
    "SemanticSnapshot",
    "SherpaOnnxStreamingASR",
    "SherpaOnnxTransducerConfig",
    "NativeProsodyExtractor",
    "SpeakerEmbedding",
    "SpectralFrames",
    "SpectralSynthesisConfig",
    "StreamingASRBackend",
    "StreamingCadenceSelector",
    "StreamingISTFT",
    "StreamingLogMelFrontend",
    "StreamingYinPitchExtractor",
    "TensorSpec",
    "TextReplayASR",
    "WhisperCppCliASR",
    "WhisperCppCliConfig",
    "TranscriptHypothesis",
    "YinPitchExtractor",
    "align_frame_conditions",
    "classify_prosody_style",
    "create_asr_backend",
    "cosine_similarity",
    "expected_mel_frames",
    "load_bundle_manifest",
    "pitch_condition_features",
    "verify_bundle",
    "waveform_to_spectral_frames",
]
__version__ = "0.4.0a2"
