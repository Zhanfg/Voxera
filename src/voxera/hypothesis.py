from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .semantic import TranscriptHypothesis


@dataclass(frozen=True, slots=True)
class HypothesisStabilizerConfig:
    confirmation_updates: int = 2
    min_partial_chars: int = 2
    max_history: int = 3

    def __post_init__(self) -> None:
        if self.confirmation_updates < 2:
            raise ValueError("confirmation_updates must be at least 2")
        if self.min_partial_chars <= 0:
            raise ValueError("min_partial_chars must be positive")
        if self.max_history < self.confirmation_updates:
            raise ValueError(
                "max_history must be at least confirmation_updates"
            )


class HypothesisStabilizer:
    """Suppress unstable ASR tails without delaying final hypotheses.

    Partial text is emitted only from the common prefix of several consecutive
    raw hypotheses. Once a prefix is published, later contradictory partials do
    not retract it; the final hypothesis always replaces the partial state.

    This trades a small amount of semantic-sidecar freshness for much lower
    intent jitter while keeping ASR entirely off the audio critical path.
    """

    def __init__(
        self,
        config: HypothesisStabilizerConfig | None = None,
    ) -> None:
        self.config = config or HypothesisStabilizerConfig()
        self._history: deque[TranscriptHypothesis] = deque(
            maxlen=self.config.max_history
        )
        self._published_text = ""

    def reset(self) -> None:
        self._history.clear()
        self._published_text = ""

    def push(
        self,
        hypothesis: TranscriptHypothesis,
    ) -> TranscriptHypothesis | None:
        if hypothesis.is_final:
            self._history.clear()
            self._published_text = ""
            return hypothesis

        self._history.append(hypothesis)
        if len(self._history) < self.config.confirmation_updates:
            return None

        recent = tuple(self._history)[-self.config.confirmation_updates :]
        candidate = _longest_common_prefix(tuple(item.text for item in recent))
        candidate = _safe_partial_prefix(candidate, tuple(item.text for item in recent))
        candidate = candidate.rstrip()

        if len(candidate) < self.config.min_partial_chars:
            return None
        if len(candidate) <= len(self._published_text):
            return None
        if self._published_text and not candidate.startswith(self._published_text):
            return None

        self._published_text = candidate
        confidence = min(item.confidence for item in recent)
        language_hint = hypothesis.language_hint
        return TranscriptHypothesis(
            text=candidate,
            confidence=confidence,
            is_final=False,
            revision=hypothesis.revision,
            language_hint=language_hint,
        )


def _longest_common_prefix(texts: tuple[str, ...]) -> str:
    if not texts:
        return ""
    prefix = texts[0]
    for text in texts[1:]:
        limit = min(len(prefix), len(text))
        index = 0
        while index < limit and prefix[index] == text[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def _safe_partial_prefix(candidate: str, texts: tuple[str, ...]) -> str:
    """Avoid publishing the middle of an ASCII word.

    Chinese/Japanese/Korean characters remain character-addressable, while a
    Latin token such as "hel" from "hello" is held until a word boundary appears.
    """

    if not candidate:
        return candidate

    last = candidate[-1]
    if not (last.isascii() and last.isalnum()):
        return candidate

    continues_ascii_word = False
    for text in texts:
        if len(text) <= len(candidate):
            continue
        next_char = text[len(candidate)]
        if next_char.isascii() and next_char.isalnum():
            continues_ascii_word = True
            break

    if not continues_ascii_word:
        return candidate

    boundary = -1
    for index, char in enumerate(candidate):
        if not (char.isascii() and char.isalnum()):
            boundary = index
    if boundary < 0:
        return ""
    return candidate[: boundary + 1]
