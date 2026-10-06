from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .semantic import TranscriptHypothesis

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SherpaOnnxTransducerConfig:
    """Files and runtime settings for a sherpa-onnx streaming transducer."""

    tokens: Path
    encoder: Path
    decoder: Path
    joiner: Path
    num_threads: int = 2
    sample_rate: int = 16_000
    feature_dim: int = 80
    decoding_method: str = "greedy_search"
    max_active_paths: int = 4
    provider: str = "cpu"
    enable_endpoint_detection: bool = True
    uncalibrated_confidence: float = 0.75
    tail_padding_seconds: float = 0.66

    def __post_init__(self) -> None:
        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.max_active_paths <= 0:
            raise ValueError("max_active_paths must be positive")
        if not 0.0 <= self.uncalibrated_confidence <= 1.0:
            raise ValueError("uncalibrated_confidence must be in [0, 1]")
        if self.tail_padding_seconds < 0.0:
            raise ValueError("tail_padding_seconds must be non-negative")

    def validate_files(self) -> None:
        for path in (self.tokens, self.encoder, self.decoder, self.joiner):
            if not Path(path).is_file():
                raise FileNotFoundError(path)


class SherpaOnnxStreamingASR:
    """Concrete StreamingASRBackend for sherpa-onnx OnlineRecognizer.

    The adapter intentionally emits only text/finality/revision. sherpa-onnx
    streaming transducer results do not expose a calibrated utterance confidence
    through the simple get_result() API, so a conservative configurable fallback
    confidence is attached and kept explicit in the configuration.
    """

    def __init__(
        self,
        config: SherpaOnnxTransducerConfig,
        *,
        recognizer: Any | None = None,
    ) -> None:
        self.config = config
        self._recognizer = recognizer or self._build_recognizer(config)
        self._stream = self._recognizer.create_stream()
        self._revision = 0
        self._last_emitted: tuple[str, bool] | None = None
        self._finished = False

    def reset(self) -> None:
        self._stream = self._recognizer.create_stream()
        self._revision = 0
        self._last_emitted = None
        self._finished = False

    def accept_audio(
        self,
        samples: FloatArray,
        sample_rate: int,
    ) -> tuple[TranscriptHypothesis, ...]:
        if self._finished:
            raise RuntimeError("ASR stream is finished; call reset() before reuse")
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("sherpa-onnx adapter expects mono 1-D audio")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains non-finite values")
        if audio.size == 0:
            return ()

        self._stream.accept_waveform(sample_rate, audio)
        return self._decode_available()

    def finish(self) -> tuple[TranscriptHypothesis, ...]:
        if self._finished:
            return ()

        tail_samples = int(
            round(self.config.tail_padding_seconds * self.config.sample_rate)
        )
        if tail_samples:
            tail = np.zeros(tail_samples, dtype=np.float32)
            self._stream.accept_waveform(self.config.sample_rate, tail)

        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        text = self._result_text()
        self._finished = True
        emitted = self._emit(text, is_final=True)
        return () if emitted is None else (emitted,)

    def _decode_available(self) -> tuple[TranscriptHypothesis, ...]:
        decoded = False
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
            decoded = True

        if not decoded:
            return ()

        text = self._result_text()
        endpoint = False
        if self.config.enable_endpoint_detection:
            endpoint = bool(self._recognizer.is_endpoint(self._stream))

        emitted = self._emit(text, is_final=endpoint)
        if endpoint:
            self._recognizer.reset(self._stream)
            self._last_emitted = None

        return () if emitted is None else (emitted,)

    def _emit(
        self,
        text: str,
        *,
        is_final: bool,
    ) -> TranscriptHypothesis | None:
        normalized = " ".join(text.strip().split())
        if not normalized:
            return None

        signature = (normalized, is_final)
        if signature == self._last_emitted:
            return None

        self._revision += 1
        self._last_emitted = signature
        return TranscriptHypothesis(
            text=normalized,
            confidence=self.config.uncalibrated_confidence,
            is_final=is_final,
            revision=self._revision,
        )

    def _result_text(self) -> str:
        result = self._recognizer.get_result(self._stream)
        if hasattr(result, "text"):
            return str(result.text)
        return str(result)

    @staticmethod
    def _build_recognizer(config: SherpaOnnxTransducerConfig) -> Any:
        config.validate_files()
        try:
            sherpa_onnx = importlib.import_module("sherpa_onnx")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "sherpa_onnx is not installed in this Python environment"
            ) from error

        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(config.tokens),
            encoder=str(config.encoder),
            decoder=str(config.decoder),
            joiner=str(config.joiner),
            num_threads=config.num_threads,
            sample_rate=config.sample_rate,
            feature_dim=config.feature_dim,
            decoding_method=config.decoding_method,
            max_active_paths=config.max_active_paths,
            provider=config.provider,
            enable_endpoint_detection=config.enable_endpoint_detection,
        )
