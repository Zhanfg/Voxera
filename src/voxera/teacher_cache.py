from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .audio import load_wav

MEANVC2_REPOSITORY = "https://github.com/ASLP-lab/MeanVC2.git"
MEANVC2_COMMIT = "13acf84c1bf135ea5edad9c245b345289b06b33e"


@dataclass(frozen=True, slots=True)
class TeacherCacheRecord:
    utterance_id: str
    wav: str
    wav_sha256: str
    source_sample_rate: int
    source_samples: int
    bn: str
    bn_frames: int
    bn_dim: int
    mel: str
    mel_frames: int
    mel_dim: int
    speaker: str
    speaker_dim: int


@dataclass(frozen=True, slots=True)
class TeacherCacheSummary:
    utterances: int
    bn_frames: int
    mel_frames: int


@dataclass(frozen=True, slots=True)
class TeacherCacheInvalidationSummary:
    stale_utterances: tuple[str, ...]
    invalidated_outputs: int
    teacher_reset: bool



def invalidate_stale_teacher_cache(
    wav_dir: Path,
    cache_dir: Path,
    *,
    manifest_path: Path | None = None,
    provenance_path: Path | None = None,
    expected_teacher_commit: str = MEANVC2_COMMIT,
) -> TeacherCacheInvalidationSummary:
    """Remove cached teacher arrays that cannot be proven fresh.

    MeanVC2's extractors intentionally skip existing .npy files. This guard runs
    before extraction so a replaced WAV can never silently reuse stale teacher
    features.
    """

    wav_dir = wav_dir.resolve()
    cache_dir = cache_dir.resolve()
    manifest_path = manifest_path or cache_dir / "manifest.jsonl"
    provenance_path = provenance_path or cache_dir / "provenance.json"

    wavs = _collect_files(wav_dir, ".wav")
    current_hashes = {
        utterance_id: _sha256(path)
        for utterance_id, path in wavs.items()
    }
    cached_ids = set()
    for subdirectory in ("bn", "mel", "speaker"):
        cached_ids.update(_collect_files(cache_dir / subdirectory, ".npy"))

    previous_hashes: dict[str, str] = {}
    manifest_trusted = False
    if manifest_path.is_file():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    utterance_id = payload["utterance_id"]
                    wav_sha256 = payload["wav_sha256"]
                    if not isinstance(utterance_id, str) or not isinstance(wav_sha256, str):
                        raise ValueError("invalid prior manifest hash record")
                    previous_hashes[utterance_id] = wav_sha256
            manifest_trusted = bool(previous_hashes)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            previous_hashes = {}
            manifest_trusted = False

    teacher_reset = False
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            teacher_reset = (
                provenance.get("teacher", {}).get("commit")
                != expected_teacher_commit
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            teacher_reset = True
    elif cached_ids:
        teacher_reset = True

    if not manifest_trusted and cached_ids:
        teacher_reset = True

    stale: set[str] = set()
    if teacher_reset:
        stale.update(cached_ids)
        stale.update(current_hashes)
    else:
        for utterance_id, current_hash in current_hashes.items():
            if previous_hashes.get(utterance_id) != current_hash:
                stale.add(utterance_id)
        stale.update(set(previous_hashes) - set(current_hashes))
        stale.update(cached_ids - set(current_hashes))

    invalidated_outputs = 0
    for utterance_id in sorted(stale):
        for subdirectory in ("bn", "mel", "speaker"):
            path = cache_dir / subdirectory / f"{utterance_id}.npy"
            if path.exists():
                path.unlink()
                invalidated_outputs += 1

    if stale or teacher_reset:
        manifest_path.unlink(missing_ok=True)
        provenance_path.unlink(missing_ok=True)

    return TeacherCacheInvalidationSummary(
        stale_utterances=tuple(sorted(stale)),
        invalidated_outputs=invalidated_outputs,
        teacher_reset=teacher_reset,
    )


def build_teacher_cache_manifest(
    wav_dir: Path,
    cache_dir: Path,
    *,
    manifest_path: Path | None = None,
    provenance_path: Path | None = None,
    teacher_repository: str = MEANVC2_REPOSITORY,
    teacher_commit: str = MEANVC2_COMMIT,
) -> TeacherCacheSummary:
    """Validate MeanVC2 preprocessing outputs and write a stable JSONL index."""

    wav_dir = wav_dir.resolve()
    cache_dir = cache_dir.resolve()
    manifest_path = manifest_path or cache_dir / "manifest.jsonl"
    provenance_path = provenance_path or cache_dir / "provenance.json"

    wavs = _collect_files(wav_dir, ".wav")
    bns = _collect_files(cache_dir / "bn", ".npy")
    mels = _collect_files(cache_dir / "mel", ".npy")
    speakers = _collect_files(cache_dir / "speaker", ".npy")

    if not wavs:
        raise ValueError(f"no WAV files found in {wav_dir}")

    expected = set(wavs)
    _require_matching_ids("bn", expected, set(bns))
    _require_matching_ids("mel", expected, set(mels))
    _require_matching_ids("speaker", expected, set(speakers))

    records: list[TeacherCacheRecord] = []
    total_bn_frames = 0
    total_mel_frames = 0

    for utterance_id in sorted(expected):
        source_audio = load_wav(wavs[utterance_id])
        if source_audio.samples.size == 0:
            raise ValueError(f"{utterance_id}: source WAV is empty")

        bn = _load_finite_array(bns[utterance_id], "BN")
        mel = _load_finite_array(mels[utterance_id], "mel")
        speaker = _load_finite_array(speakers[utterance_id], "speaker")

        if bn.ndim != 2 or bn.shape[1] != 256 or bn.shape[0] == 0:
            raise ValueError(
                f"{utterance_id}: expected BN [T, 256], got {tuple(bn.shape)}"
            )
        if mel.ndim != 2 or mel.shape[1] != 80 or mel.shape[0] == 0:
            raise ValueError(
                f"{utterance_id}: expected mel [T, 80], got {tuple(mel.shape)}"
            )
        if speaker.shape not in {(256,), (1, 256)}:
            raise ValueError(
                f"{utterance_id}: expected speaker [256], got {tuple(speaker.shape)}"
            )

        total_bn_frames += int(bn.shape[0])
        total_mel_frames += int(mel.shape[0])

        records.append(
            TeacherCacheRecord(
                utterance_id=utterance_id,
                wav=_manifest_relative(wavs[utterance_id], manifest_path.parent),
                wav_sha256=_sha256(wavs[utterance_id]),
                source_sample_rate=source_audio.sample_rate,
                source_samples=int(source_audio.samples.size),
                bn=_manifest_relative(bns[utterance_id], manifest_path.parent),
                bn_frames=int(bn.shape[0]),
                bn_dim=int(bn.shape[1]),
                mel=_manifest_relative(mels[utterance_id], manifest_path.parent),
                mel_frames=int(mel.shape[0]),
                mel_dim=int(mel.shape[1]),
                speaker=_manifest_relative(speakers[utterance_id], manifest_path.parent),
                speaker_dim=256,
            )
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    provenance = {
        "schema": 1,
        "teacher": {
            "name": "MeanVC2",
            "repository": teacher_repository,
            "commit": teacher_commit,
            "bn_extractor": "preprocess/extract_bn_80ms.py",
            "speaker_extractor": "preprocess/extract_spk_emb.py",
            "mel_extractor": "preprocess/extract_mel.py",
        },
        "contracts": {
            "bn": {"cadence_ms": 40, "dimension": 256},
            "mel": {"cadence_ms": 10, "dimension": 80},
            "speaker": {"dimension": 256},
        },
        "utterances": len(records),
        "path_base": "manifest_directory",
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return TeacherCacheSummary(
        utterances=len(records),
        bn_frames=total_bn_frames,
        mel_frames=total_mel_frames,
    )



def load_teacher_cache_manifest(manifest_path: Path) -> list[TeacherCacheRecord]:
    """Load a validated teacher manifest with paths relative to its directory."""

    manifest_path = manifest_path.resolve()
    records: list[TeacherCacheRecord] = []
    seen: set[str] = set()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                record = TeacherCacheRecord(**payload)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"{manifest_path}:{line_number}: invalid teacher-cache record"
                ) from exc

            if record.utterance_id in seen:
                raise ValueError(
                    f"{manifest_path}:{line_number}: duplicate utterance id "
                    f"{record.utterance_id}"
                )
            seen.add(record.utterance_id)
            records.append(record)

    if not records:
        raise ValueError(f"teacher manifest is empty: {manifest_path}")
    return records


def resolve_teacher_cache_path(manifest_path: Path, stored_path: str) -> Path:
    """Resolve one manifest-relative cache/source path."""

    return (manifest_path.resolve().parent / stored_path).resolve()


def _collect_files(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != suffix:
            continue
        utterance_id = path.stem
        if utterance_id in result:
            raise ValueError(f"duplicate utterance id: {utterance_id}")
        result[utterance_id] = path.resolve()
    return result


def _require_matching_ids(
    name: str,
    expected: set[str],
    actual: set[str],
) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing={missing[:8]}")
        if extra:
            pieces.append(f"extra={extra[:8]}")
        raise ValueError(f"{name} cache ids do not match WAV ids: " + ", ".join(pieces))


def _load_finite_array(path: Path, name: str) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{path}: {name} array is not numeric")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: {name} array contains non-finite values")
    return array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_relative(path: Path, manifest_dir: Path) -> str:
    return Path(os.path.relpath(path, start=manifest_dir.resolve())).as_posix()


