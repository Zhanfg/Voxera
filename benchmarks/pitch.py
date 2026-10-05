from __future__ import annotations

import time

import numpy as np

from voxera.pitch import YinPitchExtractor


def main() -> None:
    seconds = 10
    sample_rate = 16_000
    time_axis = np.arange(seconds * sample_rate, dtype=np.float32) / sample_rate
    samples = (
        0.08 * np.sin(2.0 * np.pi * 120.0 * time_axis)
        + 0.03 * np.sin(2.0 * np.pi * 240.0 * time_axis)
    ).astype(np.float32)

    extractor = YinPitchExtractor()
    started = time.perf_counter()
    track = extractor.extract(samples)
    elapsed = time.perf_counter() - started

    print(f"input_seconds={seconds}")
    print(f"frames={track.frame_count}")
    print(f"voiced_frames={int(track.voiced.sum())}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"rtf={elapsed / seconds:.6f}")


if __name__ == "__main__":
    main()
