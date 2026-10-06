from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from voxera.asr_whisper import WhisperCppCliASR, WhisperCppCliConfig


def _config() -> WhisperCppCliConfig:
    missing = Path("/not/used/in/injected-runner-tests")
    return WhisperCppCliConfig(
        binary=missing / "whisper-cli",
        model=missing / "model.bin",
        language="auto",
        use_gpu=False,
    )


def _successful_runner(command: list[str], timeout: float):
    assert timeout > 0.0
    assert "-otxt" in command
    assert "-np" in command
    assert "-nt" in command
    output_prefix = Path(command[command.index("-of") + 1])
    output_prefix.with_suffix(".txt").write_text("  你好 世界\n", encoding="utf-8")
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_accept_audio_is_non_blocking_and_finish_emits_final() -> None:
    adapter = WhisperCppCliASR(_config(), runner=_successful_runner)
    audio = np.zeros(1_600, dtype=np.float32)

    assert adapter.accept_audio(audio, 16_000) == ()
    assert adapter.accept_audio(audio, 16_000) == ()

    result = adapter.finish()

    assert len(result) == 1
    assert result[0].text == "你好 世界"
    assert result[0].is_final
    assert result[0].revision == 1
    assert result[0].language_hint is None
    assert adapter.finish() == ()


def test_reset_allows_reuse() -> None:
    adapter = WhisperCppCliASR(_config(), runner=_successful_runner)
    audio = np.zeros(800, dtype=np.float32)

    adapter.accept_audio(audio, 16_000)
    assert adapter.finish()[0].revision == 1

    with pytest.raises(RuntimeError, match="call reset"):
        adapter.accept_audio(audio, 16_000)

    adapter.reset()
    assert adapter.accept_audio(audio, 16_000) == ()
    assert adapter.finish()[0].revision == 1


def test_wrong_sample_rate_is_rejected() -> None:
    adapter = WhisperCppCliASR(_config(), runner=_successful_runner)

    with pytest.raises(ValueError, match="expects 16000 Hz"):
        adapter.accept_audio(np.zeros(80, dtype=np.float32), 8_000)


def test_cli_failure_is_propagated() -> None:
    def failing(command: list[str], timeout: float):
        del timeout
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="model load failed",
        )

    adapter = WhisperCppCliASR(_config(), runner=failing)
    adapter.accept_audio(np.zeros(160, dtype=np.float32), 16_000)

    with pytest.raises(RuntimeError, match="model load failed"):
        adapter.finish()


def test_missing_real_files_are_rejected() -> None:
    with pytest.raises(FileNotFoundError):
        WhisperCppCliASR(_config())
