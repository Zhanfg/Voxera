from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .prosody import ProsodyTrack

FloatArray = NDArray[np.float32]

SEMANTIC_FEATURE_DIM = 16
SEMANTIC_MODES = (
    "statement",
    "question",
    "command",
    "emphasis",
    "hesitation",
    "uncertainty",
    "negation",
    "continuation",
)

_ZH_QUESTION = (
    "吗",
    "呢",
    "是不是",
    "是否",
    "为什么",
    "为何",
    "怎么",
    "怎样",
    "多少",
    "谁",
    "哪里",
    "哪儿",
    "什么时候",
    "能不能",
    "可不可以",
)
_EN_QUESTION = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "is",
    "are",
    "do",
    "does",
    "did",
    "will",
    "have",
    "has",
)
_ZH_COMMAND = (
    "请",
    "务必",
    "必须",
    "别",
    "不要",
    "给我",
    "立即",
    "马上",
    "记得",
    "帮我",
)
_EN_COMMAND = (
    "please",
    "must",
    "do not",
    "don't",
    "stop",
    "go",
    "tell",
    "send",
    "make",
    "remember",
    "let",
)
_ZH_HESITATION = ("嗯", "呃", "额", "那个", "就是", "这个", "唔", "emmm")
_EN_HESITATION = ("um", "uh", "erm", "hmm", "you know", "i mean")
_ZH_UNCERTAIN = ("可能", "也许", "大概", "似乎", "好像", "不确定", "说不准")
_EN_UNCERTAIN = ("maybe", "perhaps", "probably", "seems", "not sure", "i guess")
_ZH_NEGATION = ("不", "没", "无", "别", "不是", "不能", "不会")
_EN_NEGATION = (" no ", " not ", "never", "don't", "can't", "won't", "isn't")


@dataclass(frozen=True, slots=True)
class TranscriptHypothesis:
    """One ASR hypothesis published by a sidecar backend."""

    text: str
    confidence: float = 1.0
    is_final: bool = False
    revision: int = 0
    language_hint: str | None = None

    def __post_init__(self) -> None:
        text = _normalize_text(self.text)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        if self.language_hint is not None and not self.language_hint.strip():
            raise ValueError("language_hint cannot be blank")
        object.__setattr__(self, "text", text)


@dataclass(frozen=True, slots=True)
class SemanticSnapshot:
    """Immutable, versioned semantic state consumed by the audio graph."""

    text: str
    features: FloatArray
    mode: str
    confidence: float
    language: str
    is_final: bool
    revision: int

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        if features.shape != (SEMANTIC_FEATURE_DIM,):
            raise ValueError(
                f"semantic features must have shape [{SEMANTIC_FEATURE_DIM}]"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("semantic features contain non-finite values")
        if self.mode not in SEMANTIC_MODES:
            raise ValueError(f"unsupported semantic mode: {self.mode}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        if not self.language:
            raise ValueError("language must be non-empty")
        object.__setattr__(self, "features", features)


class NativeSemanticAnalyzer:
    """Tiny deterministic language/intent analyzer.

    The analyzer is intentionally conservative. It extracts discourse cues that
    can influence pronunciation, but it does not claim to infer hidden emotion
    or sarcasm from text alone.
    """

    def analyze(
        self,
        hypothesis: TranscriptHypothesis,
        prosody: ProsodyTrack | None = None,
    ) -> SemanticSnapshot:
        text = hypothesis.text
        lower = f" {text.casefold()} "
        language, han_ratio, latin_ratio = _language_hint(
            text,
            hypothesis.language_hint,
        )

        question = _question_score(text, lower)
        command = _command_score(text, lower)
        hesitation = _phrase_score(text, lower, _ZH_HESITATION, _EN_HESITATION)
        uncertainty = _phrase_score(text, lower, _ZH_UNCERTAIN, _EN_UNCERTAIN)
        negation = _phrase_score(text, lower, _ZH_NEGATION, _EN_NEGATION)

        exclamation_marks = text.count("!") + text.count("！")
        question_marks = text.count("?") + text.count("？")
        repeated_punctuation = float(
            bool(re.search(r"([!?！？。,.，…])\1+", text))
        )
        upper_ratio = _uppercase_ratio(text)
        lexical_emphasis = float(
            any(token in text for token in ("真的", "绝对", "一定", "特别", "非常"))
            or any(
                f" {token} " in lower
                for token in ("really", "absolutely", "definitely", "very")
            )
        )

        acoustic_rise = 0.0
        acoustic_emphasis = 0.0
        acoustic_pause = 0.0
        if prosody is not None:
            acoustic_rise = float(np.clip(prosody.global_style[6], 0.0, 1.0))
            acoustic_emphasis = float(np.clip(prosody.global_style[7], 0.0, 1.0))
            acoustic_pause = float(np.clip(prosody.global_style[5], 0.0, 1.0))

        question = float(
            np.clip(
                question
                + 0.35 * float(question_marks > 0)
                + 0.20 * acoustic_rise,
                0.0,
                1.0,
            )
        )
        command = float(np.clip(command, 0.0, 1.0))
        emphasis = float(
            np.clip(
                0.35 * float(exclamation_marks > 0)
                + 0.25 * repeated_punctuation
                + 0.20 * upper_ratio
                + 0.35 * lexical_emphasis
                + 0.45 * acoustic_emphasis,
                0.0,
                1.0,
            )
        )
        hesitation = float(
            np.clip(hesitation + 0.15 * acoustic_pause, 0.0, 1.0)
        )
        uncertainty = float(np.clip(uncertainty, 0.0, 1.0))
        negation = float(np.clip(negation, 0.0, 1.0))

        continuation = float(
            np.clip(
                0.55 * float(not hypothesis.is_final)
                + 0.45 * float(_looks_continuing(text)),
                0.0,
                1.0,
            )
        )

        non_statement = max(
            question,
            command,
            emphasis,
            hesitation,
            uncertainty,
            continuation,
        )
        statement = float(np.clip(1.0 - non_statement, 0.0, 1.0))

        punctuation_count = sum(ch in "!?！？。,.，…;；:" for ch in text)
        punctuation_density = float(
            np.clip(punctuation_count / max(1, len(text)) * 8.0, 0.0, 1.0)
        )
        repetition = _repetition_score(text)
        length_norm = float(np.clip(len(text) / 80.0, 0.0, 1.0))

        features = np.asarray(
            (
                statement,
                question,
                command,
                emphasis,
                hesitation,
                uncertainty,
                negation,
                continuation,
                punctuation_density,
                repetition,
                lexical_emphasis,
                length_norm,
                han_ratio,
                latin_ratio,
                acoustic_rise,
                acoustic_emphasis,
            ),
            dtype=np.float32,
        )

        mode_scores = features[:8].copy()
        mode_index = int(np.argmax(mode_scores))
        mode = SEMANTIC_MODES[mode_index]
        ranked = np.sort(mode_scores)
        margin = float(ranked[-1] - ranked[-2]) if ranked.size > 1 else 1.0
        analyzer_confidence = float(np.clip(0.50 + 0.45 * margin, 0.50, 0.98))
        confidence = float(
            np.clip(
                hypothesis.confidence * analyzer_confidence,
                0.0,
                1.0,
            )
        )

        return SemanticSnapshot(
            text=text,
            features=features,
            mode=mode,
            confidence=confidence,
            language=language,
            is_final=hypothesis.is_final,
            revision=hypothesis.revision,
        )


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _language_hint(text: str, explicit: str | None) -> tuple[str, float, float]:
    han = sum("\u3400" <= ch <= "\u9fff" for ch in text)
    latin = sum(("a" <= ch.lower() <= "z") for ch in text)
    total = max(1, han + latin)
    han_ratio = han / total
    latin_ratio = latin / total

    if explicit:
        return explicit.strip().casefold(), float(han_ratio), float(latin_ratio)
    if han and latin:
        language = "mixed"
    elif han:
        language = "zh"
    elif latin:
        language = "en"
    else:
        language = "und"
    return language, float(han_ratio), float(latin_ratio)


def _question_score(text: str, lower: str) -> float:
    score = 0.0
    if text.endswith(("?", "？")):
        score += 0.65
    if any(token in text for token in _ZH_QUESTION):
        score += 0.55
    stripped = lower.strip()
    first = stripped.split(maxsplit=1)[0] if stripped else ""
    if first in _EN_QUESTION:
        score += 0.55
    return float(np.clip(score, 0.0, 1.0))


def _command_score(text: str, lower: str) -> float:
    score = 0.0
    if any(token in text for token in _ZH_COMMAND):
        score += 0.60
    if any(f" {token} " in lower for token in _EN_COMMAND):
        score += 0.60
    if text.endswith(("!", "！")):
        score += 0.12
    return float(np.clip(score, 0.0, 1.0))


def _phrase_score(
    text: str,
    lower: str,
    zh_tokens: tuple[str, ...],
    en_tokens: tuple[str, ...],
) -> float:
    hits = sum(token in text for token in zh_tokens)
    hits += sum(f" {token} " in lower for token in en_tokens)
    return float(np.clip(hits * 0.50, 0.0, 1.0))


def _uppercase_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha() and ch.isascii()]
    if not letters:
        return 0.0
    upper = sum(ch.isupper() for ch in letters)
    return float(upper / len(letters))


def _looks_continuing(text: str) -> bool:
    if not text:
        return True
    return text.endswith(("...", "…", ",", "，", "、", "-", "—", ":", "："))


def _repetition_score(text: str) -> float:
    if not text:
        return 0.0
    repeated_chars = len(re.findall(r"(.)\1{2,}", text))
    repeated_words = len(
        re.findall(r"\b([a-z]+)(?:\s+\1){1,}\b", text.casefold())
    )
    return float(np.clip(0.35 * repeated_chars + 0.45 * repeated_words, 0.0, 1.0))
