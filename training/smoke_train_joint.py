from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import torch

from training.encoder_training import EncoderTrainConfig, train_contentnet, train_timbrenet
from training.generator_training import (
    GeneratorTrainConfig,
    train_acoustic_generator,
    train_lite_vocoder,
)
from training.joint_training import (
    JointTrainConfig,
    checkpoint_metadata,
    find_latest_staged_checkpoints,
    train_joint_refinement,
)
from voxera.teacher_cache import MEANVC2_COMMIT, build_teacher_cache_manifest


def _write_wav(path: Path, samples: int = 6_400) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(samples, dtype=np.float32) / 16_000.0
    signal = (
        0.035 * np.sin(2.0 * np.pi * 190.0 * time)
        + 0.008 * np.sin(2.0 * np.pi * 380.0 * time)
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
    _write_wav(wav_dir / "joint.wav")

    (cache_dir / "bn").mkdir(parents=True, exist_ok=True)
    (cache_dir / "mel").mkdir(parents=True, exist_ok=True)
    (cache_dir / "speaker").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(67)
    np.save(
        cache_dir / "bn" / "joint.npy",
        rng.normal(scale=0.15, size=(8, 256)).astype(np.float32),
    )
    np.save(
        cache_dir / "mel" / "joint.npy",
        rng.normal(scale=0.15, size=(32, 80)).astype(np.float32),
    )
    np.save(
        cache_dir / "speaker" / "joint.npy",
        rng.normal(size=256).astype(np.float32),
    )

    build_teacher_cache_manifest(wav_dir, cache_dir)
    return cache_dir / "manifest.jsonl"


def main() -> int:
    torch.manual_seed(67)
    with tempfile.TemporaryDirectory(prefix="voxera-joint-smoke-") as temporary:
        root = Path(temporary)
        manifest = _build_cache(root)
        checkpoints = root / "checkpoints"
        device = torch.device("cpu")

        encoder_cfg = EncoderTrainConfig(
            learning_rate=1e-4,
            gradient_accumulation=1,
            timbre_crop_frames=24,
            seed=67,
        )
        generator_cfg = GeneratorTrainConfig(
            learning_rate=1e-4,
            gradient_accumulation=1,
            decoder_max_condition_frames=6,
            vocoder_max_frames=12,
            seed=67,
        )

        train_contentnet(
            manifest,
            checkpoints,
            epochs=1,
            device=device,
            config=encoder_cfg,
            max_records=1,
        )
        train_timbrenet(
            manifest,
            checkpoints,
            epochs=1,
            device=device,
            config=encoder_cfg,
            max_records=1,
        )
        train_acoustic_generator(
            manifest,
            checkpoints,
            epochs=1,
            device=device,
            config=generator_cfg,
            max_records=1,
        )
        train_lite_vocoder(
            manifest,
            checkpoints,
            epochs=1,
            device=device,
            config=generator_cfg,
            max_records=1,
        )

        staged = find_latest_staged_checkpoints(checkpoints)
        summaries = train_joint_refinement(
            manifest,
            staged,
            checkpoints,
            epochs=1,
            device=device,
            config=JointTrainConfig(
                learning_rate=2e-5,
                gradient_accumulation=1,
                max_condition_frames=4,
                seed=67,
            ),
            max_records=1,
        )

        if len(summaries) != 1:
            raise RuntimeError("unexpected joint-refinement summary count")
        summary = summaries[0]
        if not np.isfinite(summary.mean_loss):
            raise RuntimeError("joint-refinement loss is not finite")
        if not summary.checkpoint.is_file():
            raise RuntimeError("joint-refinement checkpoint is missing")

        metadata = checkpoint_metadata(summary.checkpoint)
        if metadata["component"] != "joint_refinement":
            raise RuntimeError("unexpected joint checkpoint component")
        if metadata["teacher"]["commit"] != MEANVC2_COMMIT:
            raise RuntimeError("joint checkpoint teacher provenance mismatch")
        if metadata["global_step"] != 1:
            raise RuntimeError("joint checkpoint step count mismatch")
        if metadata["models"] != [
            "condition_fusion",
            "contentnet",
            "decoder",
            "lite_vocoder",
            "timbrenet",
        ]:
            raise RuntimeError("joint checkpoint model set mismatch")

        print(f"joint_loss={summary.mean_loss:.6f}")
        print(f"joint_checkpoint={summary.checkpoint.name}")
        print(f"staged_contentnet={staged.contentnet.name}")
        print(f"staged_vocoder={staged.lite_vocoder.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
