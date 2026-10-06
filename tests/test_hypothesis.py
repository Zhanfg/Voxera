from __future__ import annotations

from voxera.hypothesis import HypothesisStabilizer
from voxera.semantic import TranscriptHypothesis


def _partial(text: str, revision: int) -> TranscriptHypothesis:
    return TranscriptHypothesis(
        text,
        confidence=0.9,
        is_final=False,
        revision=revision,
    )


def test_chinese_growing_partial_emits_confirmed_prefix() -> None:
    stabilizer = HypothesisStabilizer()

    assert stabilizer.push(_partial("我今天", 1)) is None
    second = stabilizer.push(_partial("我今天想", 2))
    third = stabilizer.push(_partial("我今天想继续", 3))

    assert second is not None
    assert second.text == "我今天"
    assert second.revision == 2
    assert third is not None
    assert third.text == "我今天想"


def test_latin_partial_does_not_publish_mid_word() -> None:
    stabilizer = HypothesisStabilizer()

    assert stabilizer.push(_partial("please sen", 1)) is None
    assert stabilizer.push(_partial("please send", 2)).text == "please "
    result = stabilizer.push(_partial("please send it", 3))

    assert result is not None
    assert result.text == "please send"


def test_contradictory_partial_does_not_retract_published_prefix() -> None:
    stabilizer = HypothesisStabilizer()

    stabilizer.push(_partial("今天我们", 1))
    published = stabilizer.push(_partial("今天我们继续", 2))
    assert published is not None
    assert published.text == "今天我们"

    assert stabilizer.push(_partial("今天可能暂停", 3)) is None
    assert stabilizer.push(_partial("今天可能先暂停", 4)) is None


def test_final_hypothesis_bypasses_stabilization_and_resets_phrase() -> None:
    stabilizer = HypothesisStabilizer()

    stabilizer.push(_partial("你觉得", 1))
    final = TranscriptHypothesis(
        "你觉得可以吗？",
        confidence=0.95,
        is_final=True,
        revision=2,
    )

    assert stabilizer.push(final) == final

    assert stabilizer.push(_partial("下一句", 3)) is None
    next_partial = stabilizer.push(_partial("下一句话", 4))
    assert next_partial is not None
    assert next_partial.text == "下一句"
