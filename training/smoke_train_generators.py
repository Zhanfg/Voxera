from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

from training.generator_training import (
    GeneratorTrainConfig,
    checkpoint_metadata,
    train_acoustic_generator,
    train_lite_vocoder,
)
from voxera.teacher_cache import MEANVC2_COMMIT, build_teacher_cache_manifest


def _write_wav(path: Path, samples: int = 6_400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(samples, dtype=np.float32) / 16_000.0
    signal = (
        0.04 * np.sin(2.0 * np.pi * 175.0 * time)
        + 0.01 * np.sin(2.0 * np.pi * 350.0 * time)
    )
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

    rng = np.random.default_rng(53)
    np.save(
        cache_dir / "bn" / "smoke.npy",
        rng.normal(scale=0.2, size=(8, 256)).astype(np.float32),
    )
    np.save(
        cache_dir / "mel" / "smoke.npy",
        rng.normal(scale=0.2, size=(32, 80)).astype(np.float32),
    )
    speaker = rng.normal(size=256).astype(np.float32)
    np.save(cache_dir / "speaker" / "smoke.npy", speaker)

    build_teacher_cache_manifest(wav_dir, cache_dir)
    return cache_dir / "manifest.jsonl"


def main() -> int:
    torch.manual_seed(53)
    with tempfile.TemporaryDirectory(prefix="voxera-generator-smoke-") as temporary:
        root = Path(temporary)
        manifest = _build_cache(root)
        output = root / "checkpoints"
        config = GeneratorTrainConfig(
            learning_rate=1e-4,
            gradient_accumulation=1,
            decoder_max_condition_frames=8,
            vocoder_max_frames=16,
            seed=53,
        )
        device = torch.device("cpu")

        acoustic = train_acoustic_generator(
            manifest,
            output,
            epochs=1,
            device=device,
            config=config,
            max_records=1,
        )
        vocoder = train_lite_vocoder(
            manifest,
            output,
            epochs=1,
            device=device,
            config=config,
            max_records=1,
        )

        if len(acoustic) != 1 or len(vocoder) != 1:
            raise RuntimeError("unexpected generator training summary count")
        if not np.isfinite(acoustic[0].mean_loss):
            raise RuntimeError("acoustic generator loss is not finite")
        if not np.isfinite(vocoder[0].mean_loss):
            raise RuntimeError("LiteVocoder loss is not finite")

        expected = {
            "acoustic_generator": ["condition_fusion", "decoder"],
            "lite_vocoder": ["lite_vocoder"],
        }
        for summary in (*acoustic, *vocoder):
            if not summary.checkpoint.is_file():
                raise RuntimeError(f"checkpoint missing: {summary.checkpoint}")
            metadata = checkpoint_metadata(summary.checkpoint)
            if metadata["teacher"]["commit"] != MEANVC2_COMMIT:
                raise RuntimeError("checkpoint teacher provenance mismatch")
            if metadata["epoch"] != 1 or metadata["global_step"] != 1:
                raise RuntimeError("checkpoint training counters are invalid")
            if metadata["models"] != expected[summary.component]:
                raise RuntimeError("checkpoint model set is invalid")

        print(f"acoustic_loss={acoustic[0].mean_loss:.6f}")
        print(f"vocoder_loss={vocoder[0].mean_loss:.6f}")
        print(f"acoustic_checkpoint={acoustic[0].checkpoint.name}")
        print(f"vocoder_checkpoint={vocoder[0].checkpoint.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
