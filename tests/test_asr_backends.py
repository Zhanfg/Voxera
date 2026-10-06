from __future__ import annotations

from pathlib import Path

import pytest

from voxera.asr_backends import ASRBackendKind, ASRBackendSelection
from voxera.asr_sherpa import SherpaOnnxTransducerConfig
from voxera.asr_whisper import WhisperCppCliConfig


def _sherpa() -> SherpaOnnxTransducerConfig:
    root = Path("/tmp/sherpa")
    return SherpaOnnxTransducerConfig(
        tokens=root / "tokens.txt",
        encoder=root / "encoder.onnx",
        decoder=root / "decoder.onnx",
        joiner=root / "joiner.onnx",
    )


def _whisper() -> WhisperCppCliConfig:
    root = Path("/tmp/whisper")
    return WhisperCppCliConfig(
        binary=root / "whisper-cli",
        model=root / "model.bin",
    )


def test_backend_selection_rejects_config_mismatch() -> None:
    with pytest.raises(TypeError, match="sherpa selection"):
        ASRBackendSelection(ASRBackendKind.SHERPA, _whisper())

    with pytest.raises(TypeError, match="whisper selection"):
        ASRBackendSelection(ASRBackendKind.WHISPER, _sherpa())


def test_backend_selection_accepts_matching_configs() -> None:
    assert (
        ASRBackendSelection(ASRBackendKind.SHERPA, _sherpa()).kind
        is ASRBackendKind.SHERPA
    )
    assert (
        ASRBackendSelection(ASRBackendKind.WHISPER, _whisper()).kind
        is ASRBackendKind.WHISPER
    )
