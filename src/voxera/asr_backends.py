from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .asr import StreamingASRBackend
from .asr_sherpa import SherpaOnnxStreamingASR, SherpaOnnxTransducerConfig
from .asr_whisper import WhisperCppCliASR, WhisperCppCliConfig


class ASRBackendKind(str, Enum):
    SHERPA = "sherpa"
    WHISPER = "whisper"


@dataclass(frozen=True, slots=True)
class ASRBackendSelection:
    """Explicit backend choice; no hidden auto-fallback between ASR engines."""

    kind: ASRBackendKind
    config: SherpaOnnxTransducerConfig | WhisperCppCliConfig

    def __post_init__(self) -> None:
        if self.kind is ASRBackendKind.SHERPA and not isinstance(
            self.config,
            SherpaOnnxTransducerConfig,
        ):
            raise TypeError("sherpa selection requires SherpaOnnxTransducerConfig")
        if self.kind is ASRBackendKind.WHISPER and not isinstance(
            self.config,
            WhisperCppCliConfig,
        ):
            raise TypeError("whisper selection requires WhisperCppCliConfig")


def create_asr_backend(
    selection: ASRBackendSelection,
) -> StreamingASRBackend:
    if selection.kind is ASRBackendKind.SHERPA:
        assert isinstance(selection.config, SherpaOnnxTransducerConfig)
        return SherpaOnnxStreamingASR(selection.config)
    if selection.kind is ASRBackendKind.WHISPER:
        assert isinstance(selection.config, WhisperCppCliConfig)
        return WhisperCppCliASR(selection.config)
    raise ValueError(f"unsupported ASR backend: {selection.kind}")
