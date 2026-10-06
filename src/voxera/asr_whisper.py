from __future__ import annotations

import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .semantic import TranscriptHypothesis

FloatArray = NDArray[np.float32]
Runner = Callable[[list[str], float], subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class WhisperCppCliConfig:
    """Phrase-final whisper.cpp CLI backend configuration.

    This adapter is deliberately not presented as a low-latency streaming ASR.
    It buffers source PCM and performs one offline transcription on finish().
    sherpa-onnx remains Voxera's primary partial-hypothesis streaming backend.
    """

    binary: Path
    model: Path
    language: str = "auto"
    threads: int = 2
    sample_rate: int = 16_000
    use_gpu: bool = True
    timeout_seconds: float = 120.0
    uncalibrated_confidence: float = 0.80

    def __post_init__(self) -> None:
        if self.threads <= 0:
            raise ValueError("threads must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if not 0.0 <= self.uncalibrated_confidence <= 1.0:
            raise ValueError("uncalibrated_confidence must be in [0, 1]")
        if not self.language.strip():
            raise ValueError("language must be non-empty")

    def validate_files(self) -> None:
        if not Path(self.binary).is_file():
            raise FileNotFoundError(self.binary)
        if not Path(self.model).is_file():
            raise FileNotFoundError(self.model)


class WhisperCppCliASR:
    """Phrase-final StreamingASRBackend implemented through whisper-cli.

    accept_audio() only buffers PCM and never invokes a subprocess. That keeps
    the sidecar predictable and makes it explicit that this backend is for
    phrase-final/quality fallback rather than low-latency partial decoding.
    """

    def __init__(
        self,
        config: WhisperCppCliConfig,
        *,
        runner: Runner | None = None,
    ) -> None:
        self.config = config
        if runner is None:
            config.validate_files()
            self._runner = _default_runner
        else:
            self._runner = runner
        self._chunks: list[FloatArray] = []
        self._revision = 0
        self._finished = False

    def reset(self) -> None:
        self._chunks.clear()
        self._revision = 0
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
            raise ValueError("whisper.cpp adapter expects mono 1-D audio")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains non-finite values")
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f"whisper.cpp adapter expects {self.config.sample_rate} Hz audio, "
                f"got {sample_rate}"
            )
        if audio.size:
            self._chunks.append(audio.copy())
        return ()

    def finish(self) -> tuple[TranscriptHypothesis, ...]:
        if self._finished:
            return ()
        self._finished = True

        if not self._chunks:
            return ()

        audio = np.concatenate(self._chunks).astype(np.float32, copy=False)
        text = self._transcribe(audio)
        normalized = " ".join(text.strip().split())
        if not normalized:
            return ()

        self._revision += 1
        return (
            TranscriptHypothesis(
                text=normalized,
                confidence=self.config.uncalibrated_confidence,
                is_final=True,
                revision=self._revision,
                language_hint=None
                if self.config.language == "auto"
                else self.config.language,
            ),
        )

    def _transcribe(self, audio: FloatArray) -> str:
        with tempfile.TemporaryDirectory(prefix="voxera-whisper-") as temporary:
            root = Path(temporary)
            wav_path = root / "phrase.wav"
            output_prefix = root / "transcript"
            _write_wav(wav_path, audio, self.config.sample_rate)

            command = [
                str(self.config.binary),
                "-m",
                str(self.config.model),
                "-f",
                str(wav_path),
                "-otxt",
                "-of",
                str(output_prefix),
                "-np",
                "-nt",
                "-l",
                self.config.language,
                "-t",
                str(self.config.threads),
            ]
            if not self.config.use_gpu:
                command.append("-ng")

            completed = self._runner(command, self.config.timeout_seconds)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                if len(detail) > 1_000:
                    detail = detail[-1_000:]
                raise RuntimeError(
                    "whisper-cli failed"
                    + (f": {detail}" if detail else "")
                )

            transcript_path = output_prefix.with_suffix(".txt")
            if not transcript_path.is_file():
                raise RuntimeError(
                    "whisper-cli completed without producing the expected "
                    f"transcript file: {transcript_path}"
                )
            return transcript_path.read_text(encoding="utf-8", errors="replace")


def _write_wav(path: Path, audio: FloatArray, sample_rate: int) -> None:
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _default_runner(
    command: list[str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
