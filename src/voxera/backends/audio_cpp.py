from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BackendDoctorReport:
    ready: bool
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConversionResult:
    output_path: Path
    elapsed_seconds: float
    command: tuple[str, ...]


class AudioCppMeanVC2Backend:
    """Reference MeanVC2 runner backed by audio.cpp.

    This backend exists to provide an executable M1 quality/latency oracle while
    Voxera's own inference components are implemented. It is not the final
    Voxera runtime and does not vendor audio.cpp or MeanVC2.
    """

    _VALID_BACKENDS = {"best", "cpu", "cuda", "vulkan", "metal"}

    def __init__(
        self,
        cli_path: str | os.PathLike[str],
        model_path: str | os.PathLike[str],
        *,
        compute_backend: str = "best",
    ) -> None:
        self.cli_path = Path(cli_path).expanduser()
        self.model_path = Path(model_path).expanduser()
        if compute_backend not in self._VALID_BACKENDS:
            allowed = ", ".join(sorted(self._VALID_BACKENDS))
            raise ValueError(f"compute_backend must be one of: {allowed}")
        self.compute_backend = compute_backend

    def doctor(self) -> BackendDoctorReport:
        problems: list[str] = []

        if not self.cli_path.is_file():
            problems.append(f"audio.cpp CLI not found: {self.cli_path}")
        elif os.name != "nt" and not os.access(self.cli_path, os.X_OK):
            problems.append(f"audio.cpp CLI is not executable: {self.cli_path}")

        if not self.model_path.is_file():
            problems.append(f"MeanVC2 model not found: {self.model_path}")

        return BackendDoctorReport(ready=not problems, problems=tuple(problems))

    def build_command(
        self,
        source_wav: str | os.PathLike[str],
        target_wav: str | os.PathLike[str],
        output_wav: str | os.PathLike[str],
    ) -> tuple[str, ...]:
        return (
            str(self.cli_path),
            "--task",
            "vc",
            "--family",
            "meanvc2",
            "--model",
            str(self.model_path),
            "--backend",
            self.compute_backend,
            "--mode",
            "offline",
            "--audio",
            str(Path(source_wav)),
            "--voice-ref",
            str(Path(target_wav)),
            "--out",
            str(Path(output_wav)),
            "--metrics",
        )

    def convert(
        self,
        source_wav: str | os.PathLike[str],
        target_wav: str | os.PathLike[str],
        output_wav: str | os.PathLike[str],
        *,
        timeout_seconds: float | None = None,
    ) -> ConversionResult:
        source = Path(source_wav)
        target = Path(target_wav)
        output = Path(output_wav)

        if not source.is_file():
            raise FileNotFoundError(f"source WAV not found: {source}")
        if not target.is_file():
            raise FileNotFoundError(f"target voice WAV not found: {target}")

        report = self.doctor()
        if not report.ready:
            raise RuntimeError("reference backend is not ready: " + "; ".join(report.problems))

        output.parent.mkdir(parents=True, exist_ok=True)
        command = self.build_command(source, target, output)

        started = time.perf_counter()
        completed = subprocess.run(
            command,
            check=False,
            timeout=timeout_seconds,
        )
        elapsed = time.perf_counter() - started

        if completed.returncode != 0:
            raise RuntimeError(
                f"audio.cpp MeanVC2 conversion failed with exit code {completed.returncode}"
            )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("audio.cpp returned success but produced no output WAV")

        return ConversionResult(
            output_path=output,
            elapsed_seconds=elapsed,
            command=command,
        )


def default_reference_paths(repo_root: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Return paths created by tools/bootstrap_reference.sh."""

    root = Path(repo_root)
    reference = root / ".cache" / "reference"
    cli_name = "audiocpp_cli.exe" if os.name == "nt" else "audiocpp_cli"
    cli_path = reference / "audio.cpp" / "build" / "bin" / cli_name
    model_path = (
        reference
        / "models"
        / "MeanVC2-GGUF"
        / "meanvc2-120ms-40ms-fp32.gguf"
    )
    return cli_path, model_path


def format_doctor_report(report: BackendDoctorReport) -> Sequence[str]:
    if report.ready:
        return ("reference_backend=ready",)
    return ("reference_backend=not_ready", *(f"problem={item}" for item in report.problems))
