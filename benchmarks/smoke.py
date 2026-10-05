from __future__ import annotations

import time

import numpy as np

from voxera.audio import AudioBuffer, resample_linear


def main() -> None:
    seconds = 10
    source_rate = 48_000
    target_rate = 16_000
    t = np.arange(seconds * source_rate, dtype=np.float32) / source_rate
    samples = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    audio = AudioBuffer(samples, source_rate)

    started = time.perf_counter()
    output = resample_linear(audio, target_rate)
    elapsed = time.perf_counter() - started
    rtf = elapsed / seconds
    print(f"input_seconds={seconds}")
    print(f"output_samples={output.samples.size}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"rtf={rtf:.6f}")


if __name__ == "__main__":
    main()
