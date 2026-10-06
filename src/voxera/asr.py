from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .hypothesis import HypothesisStabilizer
from .prosody import ProsodyTrack
from .semantic import (
    NativeSemanticAnalyzer,
    SemanticSnapshot,
    TranscriptHypothesis,
)

FloatArray = NDArray[np.float32]


class StreamingASRBackend(Protocol):
    """Backend-neutral contract for sherpa-onnx, whisper.cpp, or another ASR."""

    def reset(self) -> None:
        ...

    def accept_audio(
        self,
        samples: FloatArray,
        sample_rate: int,
    ) -> tuple[TranscriptHypothesis, ...]:
        ...

    def finish(self) -> tuple[TranscriptHypothesis, ...]:
        ...


@dataclass(slots=True)
class SemanticMailbox:
    """Single-producer/latest-value mailbox for the audio graph.

    The audio path only reads immutable snapshots. It never calls ASR or waits
    for transcript generation.
    """

    _latest: SemanticSnapshot | None = None

    def publish(self, snapshot: SemanticSnapshot) -> None:
        current = self._latest
        if current is not None and snapshot.revision < current.revision:
            raise ValueError("semantic revisions must not move backwards")
        self._latest = snapshot

    def read_latest(self) -> SemanticSnapshot | None:
        return self._latest

    def clear(self) -> None:
        self._latest = None


class SemanticSidecar:
    """Compose a streaming ASR backend and semantic analyzer.

    Hosts should run this object on a background worker. The conversion path
    consumes only SemanticMailbox.read_latest(), keeping ASR off the real-time
    critical path.
    """

    def __init__(
        self,
        backend: StreamingASRBackend,
        *,
        analyzer: NativeSemanticAnalyzer | None = None,
        mailbox: SemanticMailbox | None = None,
        stabilizer: HypothesisStabilizer | None = None,
    ) -> None:
        self.backend = backend
        self.analyzer = analyzer or NativeSemanticAnalyzer()
        self.mailbox = mailbox or SemanticMailbox()
        self.stabilizer = stabilizer

    def reset(self) -> None:
        self.backend.reset()
        self.mailbox.clear()
        if self.stabilizer is not None:
            self.stabilizer.reset()

    def push_audio(
        self,
        samples: FloatArray,
        sample_rate: int,
        *,
        prosody: ProsodyTrack | None = None,
    ) -> tuple[SemanticSnapshot, ...]:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("ASR sidecar expects mono 1-D audio")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not np.all(np.isfinite(audio)):
            raise ValueError("audio contains non-finite values")

        return self._consume(self.backend.accept_audio(audio, sample_rate), prosody)

    def finish(
        self,
        *,
        prosody: ProsodyTrack | None = None,
    ) -> tuple[SemanticSnapshot, ...]:
        return self._consume(self.backend.finish(), prosody)

    def _consume(
        self,
        hypotheses: tuple[TranscriptHypothesis, ...],
        prosody: ProsodyTrack | None,
    ) -> tuple[SemanticSnapshot, ...]:
        snapshots: list[SemanticSnapshot] = []
        for hypothesis in hypotheses:
            if self.stabilizer is not None:
                stabilized = self.stabilizer.push(hypothesis)
                if stabilized is None:
                    continue
                hypothesis = stabilized

            snapshot = self.analyzer.analyze(hypothesis, prosody)
            self.mailbox.publish(snapshot)
            snapshots.append(snapshot)
        return tuple(snapshots)


class TextReplayASR:
    """Tiny deterministic ASR stub for tests and integration development."""

    def __init__(self, hypotheses: tuple[TranscriptHypothesis, ...]) -> None:
        self._hypotheses = hypotheses
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def accept_audio(
        self,
        samples: FloatArray,
        sample_rate: int,
    ) -> tuple[TranscriptHypothesis, ...]:
        del samples, sample_rate
        if self._cursor >= len(self._hypotheses):
            return ()
        hypothesis = self._hypotheses[self._cursor]
        self._cursor += 1
        return (hypothesis,)

    def finish(self) -> tuple[TranscriptHypothesis, ...]:
        if self._cursor >= len(self._hypotheses):
            return ()
        remaining = self._hypotheses[self._cursor :]
        self._cursor = len(self._hypotheses)
        return remaining
