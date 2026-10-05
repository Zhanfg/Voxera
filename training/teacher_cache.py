from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


MEANVC2_REPOSITORY = "https://github.com/ASLP-lab/MeanVC2.git"
MEANVC2_COMMIT = "13acf84c1bf135ea5edad9c245b345289b06b33e"


@dataclass(frozen=True, slots=True)
class TeacherCacheRecord:
    utterance_id: str
    wav: str
    wav_sha256: str
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
    return Path(
        __import__("os").path.relpath(path, start=manifest_dir.resolve())
    ).as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate MeanVC2 teacher outputs and build Voxera manifest"
    )
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_teacher_cache_manifest(args.wav_dir, args.cache_dir)
    print(f"utterances={summary.utterances}")
    print(f"bn_frames={summary.bn_frames}")
    print(f"mel_frames={summary.mel_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
