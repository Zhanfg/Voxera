from __future__ import annotations

import numpy as np
import pytest

from voxera.asr import SemanticMailbox, SemanticSidecar, TextReplayASR
from voxera.hypothesis import HypothesisStabilizer
from voxera.semantic import NativeSemanticAnalyzer, TranscriptHypothesis


def test_chinese_question_is_detected() -> None:
    snapshot = NativeSemanticAnalyzer().analyze(
        TranscriptHypothesis(
            "你今天还会继续做吗？",
            confidence=0.95,
            is_final=True,
            revision=1,
        )
    )

    assert snapshot.mode == "question"
    assert snapshot.language == "zh"
    assert snapshot.features[1] > snapshot.features[0]
    assert snapshot.is_final
    assert snapshot.revision == 1


def test_english_command_is_detected() -> None:
    snapshot = NativeSemanticAnalyzer().analyze(
        TranscriptHypothesis(
            "Please send the file now.",
            confidence=0.9,
            is_final=True,
            revision=2,
        )
    )

    assert snapshot.mode == "command"
    assert snapshot.language == "en"
    assert snapshot.features[2] >= 0.5


def test_partial_hypothesis_marks_continuation() -> None:
    snapshot = NativeSemanticAnalyzer().analyze(
        TranscriptHypothesis(
            "我觉得可能还要",
            confidence=0.8,
            is_final=False,
            revision=3,
        )
    )

    assert snapshot.features[7] >= 0.5
    assert snapshot.mode in {"continuation", "uncertainty"}


def test_mixed_language_is_reported() -> None:
    snapshot = NativeSemanticAnalyzer().analyze(
        TranscriptHypothesis(
            "这个 model 可以吗？",
            confidence=1.0,
            is_final=True,
            revision=4,
        )
    )

    assert snapshot.language == "mixed"
    assert snapshot.features[12] > 0.0
    assert snapshot.features[13] > 0.0


def test_mailbox_rejects_revision_regression() -> None:
    analyzer = NativeSemanticAnalyzer()
    mailbox = SemanticMailbox()

    newer = analyzer.analyze(
        TranscriptHypothesis("好的。", is_final=True, revision=5)
    )
    older = analyzer.analyze(
        TranscriptHypothesis("旧结果。", is_final=True, revision=4)
    )
    mailbox.publish(newer)

    with pytest.raises(ValueError, match="move backwards"):
        mailbox.publish(older)


def test_sidecar_publishes_latest_without_audio_path_asr_dependency() -> None:
    backend = TextReplayASR(
        (
            TranscriptHypothesis("你确定吗", confidence=0.8, revision=1),
            TranscriptHypothesis(
                "你确定吗？",
                confidence=0.95,
                is_final=True,
                revision=2,
            ),
        )
    )
    sidecar = SemanticSidecar(backend)
    audio = np.zeros(320, dtype=np.float32)

    first = sidecar.push_audio(audio, 16_000)
    second = sidecar.push_audio(audio, 16_000)

    assert len(first) == 1
    assert len(second) == 1
    assert sidecar.mailbox.read_latest() == second[0]
    assert second[0].revision == 2
    assert second[0].mode == "question"



def test_sidecar_can_stabilize_partial_hypotheses_before_semantics() -> None:
    backend = TextReplayASR(
        (
            TranscriptHypothesis("我今天", confidence=0.9, revision=1),
            TranscriptHypothesis("我今天想", confidence=0.9, revision=2),
            TranscriptHypothesis(
                "我今天想继续。",
                confidence=0.95,
                is_final=True,
                revision=3,
            ),
        )
    )
    sidecar = SemanticSidecar(
        backend,
        stabilizer=HypothesisStabilizer(),
    )
    audio = np.zeros(320, dtype=np.float32)

    assert sidecar.push_audio(audio, 16_000) == ()

    stable = sidecar.push_audio(audio, 16_000)
    assert len(stable) == 1
    assert stable[0].text == "我今天"
    assert stable[0].revision == 2
    assert not stable[0].is_final

    final = sidecar.push_audio(audio, 16_000)
    assert len(final) == 1
    assert final[0].text == "我今天想继续。"
    assert final[0].is_final
    assert final[0].revision == 3


def test_sidecar_reset_resets_stabilizer_state() -> None:
    hypotheses = (
        TranscriptHypothesis("第一句", confidence=0.9, revision=1),
        TranscriptHypothesis("第一句话", confidence=0.9, revision=2),
    )
    sidecar = SemanticSidecar(
        TextReplayASR(hypotheses),
        stabilizer=HypothesisStabilizer(),
    )
    audio = np.zeros(320, dtype=np.float32)

    assert sidecar.push_audio(audio, 16_000) == ()
    assert len(sidecar.push_audio(audio, 16_000)) == 1

    sidecar.reset()
    assert sidecar.mailbox.read_latest() is None
    assert sidecar.push_audio(audio, 16_000) == ()
