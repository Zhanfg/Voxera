from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from voxera.audio import load_wav


def audit_wav_directory(wav_dir: Path) -> dict[str, object]:
    wav_dir = wav_dir.resolve()
    wavs = sorted(path for path in wav_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav")
    if not wavs:
        raise ValueError(f"no WAV files found in {wav_dir}")

    total_seconds = 0.0
    min_seconds = float("inf")
    max_seconds = 0.0
    short_count = 0
    sample_rates: Counter[int] = Counter()

    for path in wavs:
        audio = load_wav(path)
        if audio.samples.size == 0:
            raise ValueError(f"empty WAV: {path}")
        duration = audio.duration_seconds
        total_seconds += duration
        min_seconds = min(min_seconds, duration)
        max_seconds = max(max_seconds, duration)
        sample_rates[audio.sample_rate] += 1
        if duration < 0.4:
            short_count += 1

    return {
        "files": len(wavs),
        "total_seconds": total_seconds,
        "total_hours": total_seconds / 3600.0,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "under_400ms": short_count,
        "sample_rates": dict(sorted(sample_rates.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Voxera M1 training WAVs")
    parser.add_argument("--wav-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = audit_wav_directory(args.wav_dir)

    print(f"files={summary['files']}")
    print(f"total_hours={summary['total_hours']:.4f}")
    print(f"min_seconds={summary['min_seconds']:.4f}")
    print(f"max_seconds={summary['max_seconds']:.4f}")
    print(f"under_400ms={summary['under_400ms']}")
    print(f"sample_rates={summary['sample_rates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
