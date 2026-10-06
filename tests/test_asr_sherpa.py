from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxera.asr_sherpa import SherpaOnnxStreamingASR, SherpaOnnxTransducerConfig


class FakeStream:
    def __init__(self) -> None:
        self.chunks = 0
        self.decoded = 0
        self.finished = False

    def accept_waveform(self, sample_rate, samples) -> None:
        assert sample_rate > 0
        assert samples.ndim == 1
        self.chunks += 1

    def input_finished(self) -> None:
        self.finished = True


class FakeRecognizer:
    def __init__(self) -> None:
        self.reset_calls = 0

    def create_stream(self):
        return FakeStream()

    def is_ready(self, stream) -> bool:
        return stream.decoded < stream.chunks

    def decode_stream(self, stream) -> None:
        stream.decoded += 1

    def get_result(self, stream):
        if stream.finished:
            return "最终文本"
        if stream.chunks >= 2:
            return "你好吗？"
        return "你好吗"

    def is_endpoint(self, stream) -> bool:
        return stream.chunks >= 2 and not stream.finished

    def reset(self, stream) -> None:
        self.reset_calls += 1
        stream.chunks = 0
        stream.decoded = 0
        stream.finished = False


def _config() -> SherpaOnnxTransducerConfig:
    missing = Path("/not/used/in/injected-recognizer-tests")
    return SherpaOnnxTransducerConfig(
        tokens=missing / "tokens.txt",
        encoder=missing / "encoder.onnx",
        decoder=missing / "decoder.onnx",
        joiner=missing / "joiner.onnx",
    )


def test_partial_then_endpoint_final_revision() -> None:
    recognizer = FakeRecognizer()
    adapter = SherpaOnnxStreamingASR(_config(), recognizer=recognizer)
    audio = np.zeros(1_600, dtype=np.float32)

    first = adapter.accept_audio(audio, 16_000)
    second = adapter.accept_audio(audio, 16_000)

    assert len(first) == 1
    assert first[0].text == "你好吗"
    assert not first[0].is_final
    assert first[0].revision == 1

    assert len(second) == 1
    assert second[0].text == "你好吗？"
    assert second[0].is_final
    assert second[0].revision == 2
    assert recognizer.reset_calls == 1


def test_finish_flushes_and_marks_final() -> None:
    adapter = SherpaOnnxStreamingASR(_config(), recognizer=FakeRecognizer())
    audio = np.zeros(800, dtype=np.float32)

    partial = adapter.accept_audio(audio, 16_000)
    final = adapter.finish()

    assert len(partial) == 1
    assert len(final) == 1
    assert final[0].text == "最终文本"
    assert final[0].is_final
    assert final[0].revision == 2
    assert adapter.finish() == ()


def test_reset_allows_reuse_and_restarts_revision() -> None:
    adapter = SherpaOnnxStreamingASR(_config(), recognizer=FakeRecognizer())
    audio = np.zeros(800, dtype=np.float32)

    assert adapter.accept_audio(audio, 16_000)[0].revision == 1
    adapter.finish()

    with pytest.raises(RuntimeError, match="call reset"):
        adapter.accept_audio(audio, 16_000)

    adapter.reset()
    assert adapter.accept_audio(audio, 16_000)[0].revision == 1


def test_real_construction_requires_model_files() -> None:
    with pytest.raises(FileNotFoundError):
        SherpaOnnxStreamingASR(_config())
