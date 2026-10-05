from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

from training.encoder_training import (
    EncoderTrainConfig,
    checkpoint_metadata,
    train_contentnet,
    train_timbrenet,
)
from voxera.teacher_cache import MEANVC2_COMMIT, build_teacher_cache_manifest


def _write_wav(path: Path, samples: int = 8_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(samples, dtype=np.float32) / 16_000.0
    signal = 0.05 * np.sin(2.0 * np.pi * 180.0 * time)
    pcm = np.rint(np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())


def _build_cache(root: Path) -> Path:
    wav_dir = root / "data" / "train_wavs"
    cache_dir = root / "teacher"
    _write_wav(wav_dir / "smoke.wav")

    (cache_dir / "bn").mkdir(parents=True, exist_ok=True)
    (cache_dir / "mel").mkdir(parents=True, exist_ok=True)
    (cache_dir / "speaker").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(41)
    np.save(
        cache_dir / "bn" / "smoke.npy",
        rng.normal(size=(10, 256)).astype(np.float32),
    )
    np.save(
        cache_dir / "mel" / "smoke.npy",
        rng.normal(size=(50, 80)).astype(np.float32),
    )
    speaker = rng.normal(size=256).astype(np.float32)
    np.save(cache_dir / "speaker" / "smoke.npy", speaker)

    build_teacher_cache_manifest(wav_dir, cache_dir)
    return cache_dir / "manifest.jsonl"


def main() -> int:
    torch.manual_seed(41)
    with tempfile.TemporaryDirectory(prefix="voxera-train-smoke-") as temporary:
        root = Path(temporary)
        manifest = _build_cache(root)
        output = root / "checkpoints"
        config = EncoderTrainConfig(
            learning_rate=1e-4,
            gradient_accumulation=1,
            timbre_crop_frames=32,
            seed=41,
        )
        device = torch.device("cpu")

        content = train_contentnet(
            manifest,
            output,
            epochs=1,
            device=device,
            config=config,
            max_records=1,
        )
        timbre = train_timbrenet(
            manifest,
            output,
            epochs=1,
            device=device,
            config=config,
            max_records=1,
        )

        if len(content) != 1 or len(timbre) != 1:
            raise RuntimeError("unexpected training summary count")
        if not np.isfinite(content[0].mean_loss):
            raise RuntimeError("ContentNet training loss is not finite")
        if not np.isfinite(timbre[0].mean_loss):
            raise RuntimeError("TimbreNet training loss is not finite")

        for summary in (*content, *timbre):
            if not summary.checkpoint.is_file():
                raise RuntimeError(f"checkpoint missing: {summary.checkpoint}")
            metadata = checkpoint_metadata(summary.checkpoint)
            if metadata["teacher"]["commit"] != MEANVC2_COMMIT:
                raise RuntimeError("checkpoint teacher provenance mismatch")
            if metadata["epoch"] != 1 or metadata["global_step"] != 1:
                raise RuntimeError("checkpoint training counters are invalid")

        print(f"content_loss={content[0].mean_loss:.6f}")
        print(f"timbre_loss={timbre[0].mean_loss:.6f}")
        print(f"content_checkpoint={content[0].checkpoint.name}")
        print(f"timbre_checkpoint={timbre[0].checkpoint.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
