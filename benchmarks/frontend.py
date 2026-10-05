from __future__ import annotations

import time

import numpy as np

from voxera.features import LogMelFrontend


def main() -> None:
    seconds = 60
    sample_rate = 16_000
    time_axis = np.arange(seconds * sample_rate, dtype=np.float32) / sample_rate
    samples = (
        0.08 * np.sin(2.0 * np.pi * 180.0 * time_axis)
        + 0.03 * np.sin(2.0 * np.pi * 1_200.0 * time_axis)
    ).astype(np.float32)

    frontend = LogMelFrontend()
    started = time.perf_counter()
    features = frontend.extract(samples)
    elapsed = time.perf_counter() - started

    print(f"input_seconds={seconds}")
    print(f"frames={features.frame_count}")
    print(f"feature_dim={features.feature_dim}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"rtf={elapsed / seconds:.6f}")


if __name__ == "__main__":
    main()
