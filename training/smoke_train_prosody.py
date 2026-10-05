from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

from training.prosody_training import (
    ProsodyTrainConfig,
    checkpoint_metadata,
    train_prosodynet,
)


def _write_wav(path: Path, samples: int = 9_600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(samples, dtype=np.float32) / 16_000.0
    frequency = 155.0 + 45.0 * np.clip((t - 0.38) / 0.22, 0.0, 1.0)
    phase = 2.0 * np.pi * np.cumsum(frequency) / 16_000.0
    envelope = np.full(samples, 0.035, dtype=np.float32)
    envelope[4_000:6_000] = 0.12
    signal = envelope * np.sin(phase)
    pcm = np.rint(np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())


def main() -> int:
    torch.manual_seed(83)
    with tempfile.TemporaryDirectory(prefix="voxera-prosody-smoke-") as temporary:
        root = Path(temporary)
        wav_dir = root / "wavs"
        _write_wav(wav_dir / "delivery.wav")

        summaries = train_prosodynet(
            wav_dir,
            root / "checkpoints",
            epochs=1,
            device=torch.device("cpu"),
            config=ProsodyTrainConfig(
                learning_rate=1e-4,
                gradient_accumulation=1,
                max_dense_frames=256,
                seed=83,
            ),
            max_files=1,
        )
        if len(summaries) != 1:
            raise RuntimeError("unexpected ProsodyNet training summary count")

        summary = summaries[0]
        if not np.isfinite(summary.mean_loss):
            raise RuntimeError("ProsodyNet loss is not finite")
        if not summary.checkpoint.is_file():
            raise RuntimeError("ProsodyNet checkpoint is missing")

        metadata = checkpoint_metadata(summary.checkpoint)
        if metadata["component"] != "prosodynet":
            raise RuntimeError("unexpected ProsodyNet checkpoint component")
        if metadata["epoch"] != 1 or metadata["global_step"] != 1:
            raise RuntimeError("ProsodyNet checkpoint counters are invalid")
        if len(metadata["dataset"]["sha256"]) != 64:
            raise RuntimeError("ProsodyNet dataset fingerprint is invalid")

        print(f"prosody_loss={summary.mean_loss:.6f}")
        print(f"checkpoint={summary.checkpoint.name}")
        print(f"dataset_sha256={metadata['dataset']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
