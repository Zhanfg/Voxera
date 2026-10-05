from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from voxera.teacher_cache import (
    MEANVC2_COMMIT,
    build_teacher_cache_manifest,
    invalidate_stale_teacher_cache,
)


def _write_wav(path: Path, samples: int = 1_600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros(samples, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(data.tobytes())


def _valid_cache(tmp_path: Path) -> tuple[Path, Path]:
    wav_dir = tmp_path / "data" / "train_wavs"
    cache_dir = tmp_path / "teacher"
    _write_wav(wav_dir / "utt_a.wav")
    _write_wav(wav_dir / "utt_b.wav")

    for uid in ("utt_a", "utt_b"):
        (cache_dir / "bn").mkdir(parents=True, exist_ok=True)
        (cache_dir / "mel").mkdir(parents=True, exist_ok=True)
        (cache_dir / "speaker").mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "bn" / f"{uid}.npy", np.zeros((5, 256), dtype=np.float32))
        np.save(cache_dir / "mel" / f"{uid}.npy", np.zeros((20, 80), dtype=np.float32))
        np.save(cache_dir / "speaker" / f"{uid}.npy", np.ones(256, dtype=np.float32))

    return wav_dir, cache_dir


def test_manifest_records_valid_teacher_contract(tmp_path: Path) -> None:
    wav_dir, cache_dir = _valid_cache(tmp_path)

    summary = build_teacher_cache_manifest(wav_dir, cache_dir)

    assert summary.utterances == 2
    assert summary.bn_frames == 10
    assert summary.mel_frames == 40

    rows = [
        json.loads(line)
        for line in (cache_dir / "manifest.jsonl").read_text().splitlines()
    ]
    assert [row["utterance_id"] for row in rows] == ["utt_a", "utt_b"]
    assert all(row["bn_dim"] == 256 for row in rows)
    assert all(row["mel_dim"] == 80 for row in rows)
    assert all(row["speaker_dim"] == 256 for row in rows)
    assert all(row["source_sample_rate"] == 16_000 for row in rows)
    assert all(row["source_samples"] == 1_600 for row in rows)
    assert all(len(row["wav_sha256"]) == 64 for row in rows)
    assert all((cache_dir / row["wav"]).resolve().is_file() for row in rows)
    assert all((cache_dir / row["bn"]).resolve().is_file() for row in rows)
    assert all((cache_dir / row["mel"]).resolve().is_file() for row in rows)
    assert all((cache_dir / row["speaker"]).resolve().is_file() for row in rows)

    provenance = json.loads((cache_dir / "provenance.json").read_text())
    assert provenance["teacher"]["commit"] == MEANVC2_COMMIT
    assert provenance["contracts"]["bn"]["cadence_ms"] == 40
    assert provenance["path_base"] == "manifest_directory"


def test_missing_teacher_output_is_rejected(tmp_path: Path) -> None:
    wav_dir, cache_dir = _valid_cache(tmp_path)
    (cache_dir / "bn" / "utt_b.npy").unlink()

    with pytest.raises(ValueError, match="bn cache ids do not match"):
        build_teacher_cache_manifest(wav_dir, cache_dir)


@pytest.mark.parametrize(
    ("kind", "shape"),
    [
        ("bn", (5, 255)),
        ("mel", (20, 79)),
        ("speaker", (255,)),
    ],
)
def test_invalid_teacher_shapes_are_rejected(
    tmp_path: Path,
    kind: str,
    shape: tuple[int, ...],
) -> None:
    wav_dir, cache_dir = _valid_cache(tmp_path)
    np.save(cache_dir / kind / "utt_a.npy", np.zeros(shape, dtype=np.float32))

    with pytest.raises(ValueError):
        build_teacher_cache_manifest(wav_dir, cache_dir)


def test_non_finite_teacher_output_is_rejected(tmp_path: Path) -> None:
    wav_dir, cache_dir = _valid_cache(tmp_path)
    bn = np.zeros((5, 256), dtype=np.float32)
    bn[0, 0] = np.nan
    np.save(cache_dir / "bn" / "utt_a.npy", bn)

    with pytest.raises(ValueError, match="non-finite"):
        build_teacher_cache_manifest(wav_dir, cache_dir)



def test_changed_wav_invalidates_only_affected_teacher_outputs(tmp_path: Path) -> None:
    wav_dir, cache_dir = _valid_cache(tmp_path)
    build_teacher_cache_manifest(wav_dir, cache_dir)

    _write_wav(wav_dir / "utt_a.wav", samples=2_000)
    summary = invalidate_stale_teacher_cache(wav_dir, cache_dir)

    assert summary.stale_utterances == ("utt_a",)
    assert summary.invalidated_outputs == 3
    assert not summary.teacher_reset
    assert not (cache_dir / "bn" / "utt_a.npy").exists()
    assert not (cache_dir / "mel" / "utt_a.npy").exists()
    assert not (cache_dir / "speaker" / "utt_a.npy").exists()
    assert (cache_dir / "bn" / "utt_b.npy").is_file()
    assert not (cache_dir / "manifest.jsonl").exists()
    assert not (cache_dir / "provenance.json").exists()


def test_unchanged_cache_remains_valid(tmp_path: Path) -> None:
    wav_dir, cache_dir = _valid_cache(tmp_path)
    build_teacher_cache_manifest(wav_dir, cache_dir)

    summary = invalidate_stale_teacher_cache(wav_dir, cache_dir)

    assert summary.stale_utterances == ()
    assert summary.invalidated_outputs == 0
    assert not summary.teacher_reset
    assert (cache_dir / "manifest.jsonl").is_file()
    assert (cache_dir / "provenance.json").is_file()


def test_removed_wav_cleans_orphan_teacher_outputs(tmp_path: Path) -> None:
    wav_dir, cache_dir = _valid_cache(tmp_path)
    build_teacher_cache_manifest(wav_dir, cache_dir)

    (wav_dir / "utt_b.wav").unlink()
    summary = invalidate_stale_teacher_cache(wav_dir, cache_dir)

    assert summary.stale_utterances == ("utt_b",)
    assert summary.invalidated_outputs == 3
    assert not (cache_dir / "bn" / "utt_b.npy").exists()
    assert (cache_dir / "bn" / "utt_a.npy").is_file()


def test_teacher_commit_change_invalidates_entire_cache(tmp_path: Path) -> None:
    wav_dir, cache_dir = _valid_cache(tmp_path)
    build_teacher_cache_manifest(wav_dir, cache_dir)

    provenance_path = cache_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["teacher"]["commit"] = "different-teacher-commit"
    provenance_path.write_text(json.dumps(provenance))

    summary = invalidate_stale_teacher_cache(wav_dir, cache_dir)

    assert summary.teacher_reset
    assert summary.stale_utterances == ("utt_a", "utt_b")
    assert summary.invalidated_outputs == 6
    assert not (cache_dir / "manifest.jsonl").exists()
    assert not (cache_dir / "provenance.json").exists()
