from __future__ import annotations

import os
from pathlib import Path

import pytest

from voxera.backends.audio_cpp import AudioCppMeanVC2Backend


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_doctor_reports_missing_runtime(tmp_path: Path) -> None:
    backend = AudioCppMeanVC2Backend(
        tmp_path / "missing-cli",
        tmp_path / "missing-model.gguf",
    )
    report = backend.doctor()
    assert not report.ready
    assert len(report.problems) == 2


def test_build_command_is_shell_free_and_explicit(tmp_path: Path) -> None:
    cli = _touch(tmp_path / "audiocpp_cli")
    if os.name != "nt":
        cli.chmod(0o755)
    model = _touch(tmp_path / "model.gguf")
    backend = AudioCppMeanVC2Backend(cli, model, compute_backend="cpu")

    command = backend.build_command("source.wav", "target.wav", "output.wav")

    assert command[0] == str(cli)
    assert "--family" in command
    assert command[command.index("--family") + 1] == "meanvc2"
    assert command[command.index("--backend") + 1] == "cpu"
    assert command[command.index("--audio") + 1] == "source.wav"
    assert command[command.index("--voice-ref") + 1] == "target.wav"


def test_convert_validates_and_returns_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _touch(tmp_path / "audiocpp_cli")
    if os.name != "nt":
        cli.chmod(0o755)
    model = _touch(tmp_path / "model.gguf")
    source = _touch(tmp_path / "source.wav")
    target = _touch(tmp_path / "target.wav")
    output = tmp_path / "out" / "converted.wav"
    backend = AudioCppMeanVC2Backend(cli, model)

    class Result:
        returncode = 0

    def fake_run(command, *, check, timeout):
        assert check is False
        assert timeout == 30
        assert tuple(command) == backend.build_command(source, target, output)
        _touch(output, b"RIFF-test")
        return Result()

    monkeypatch.setattr("voxera.backends.audio_cpp.subprocess.run", fake_run)
    result = backend.convert(source, target, output, timeout_seconds=30)

    assert result.output_path == output
    assert output.read_bytes() == b"RIFF-test"
    assert result.elapsed_seconds >= 0.0


def test_invalid_compute_backend_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        AudioCppMeanVC2Backend(tmp_path / "cli", tmp_path / "model", compute_backend="tpu")
