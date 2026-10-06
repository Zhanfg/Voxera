from __future__ import annotations

from pathlib import Path

import pytest

from voxera.deployment import (
    BundleManifest,
    ModelArtifact,
    PrecisionPolicy,
    TensorSpec,
    load_bundle_manifest,
    sha256_file,
    verify_bundle,
)


def _artifact(path: Path) -> ModelArtifact:
    return ModelArtifact(
        name="contentnet",
        file=path.name,
        sha256=sha256_file(path),
        opset=17,
        required=True,
        streaming=True,
        inputs=(
            TensorSpec("features", "float32", ("batch", "frames", 80)),
            TensorSpec("cache", "float32", (12, "batch", 64, 192)),
        ),
        outputs=(
            TensorSpec("content", "float32", ("batch", "frames", 256)),
            TensorSpec("next_cache", "float32", (12, "batch", 64, 192)),
        ),
        precision=PrecisionPolicy(),
    )


def test_manifest_roundtrip_and_bundle_verification(tmp_path: Path) -> None:
    model = tmp_path / "contentnet.onnx"
    model.write_bytes(b"fake-onnx-for-contract-test")
    manifest = BundleManifest(
        voxera_version="0.test",
        weights_status="unit-test",
        sample_rate=16_000,
        acoustic_hop_ms=10,
        condition_hop_ms=40,
        artifacts=(_artifact(model),),
    )
    path = tmp_path / "bundle.json"

    manifest.write(path)
    loaded = load_bundle_manifest(path)
    verify_bundle(tmp_path, loaded)

    assert loaded == manifest
    assert loaded.artifacts[0].inputs[0].shape == ("batch", "frames", 80)


def test_bundle_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "contentnet.onnx"
    model.write_bytes(b"first")
    manifest = BundleManifest(
        voxera_version="0.test",
        weights_status="unit-test",
        sample_rate=16_000,
        acoustic_hop_ms=10,
        condition_hop_ms=40,
        artifacts=(_artifact(model),),
    )
    model.write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_bundle(tmp_path, manifest)


def test_invalid_precision_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported precision"):
        PrecisionPolicy(candidates=("int3",))
