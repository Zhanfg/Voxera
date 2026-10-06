from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEPLOYMENT_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int | str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must be non-empty")
        if not self.dtype:
            raise ValueError("tensor dtype must be non-empty")
        shape = tuple(self.shape)
        if not shape:
            raise ValueError("tensor shape must be non-empty")
        for dimension in shape:
            if isinstance(dimension, int) and dimension <= 0:
                raise ValueError("static tensor dimensions must be positive")
            if isinstance(dimension, str) and not dimension:
                raise ValueError("dynamic tensor dimensions must be named")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    baseline: str = "fp32"
    candidates: tuple[str, ...] = ("fp16", "int8_static")
    calibration_required: tuple[str, ...] = ("int8_static",)

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        calibration_required = tuple(self.calibration_required)
        allowed = {"fp32", "fp16", "int8_dynamic", "int8_static", "q4_weight_only"}
        values = (self.baseline, *candidates, *calibration_required)
        if any(value not in allowed for value in values):
            raise ValueError("unsupported precision policy value")
        if self.baseline in calibration_required:
            raise ValueError("baseline precision cannot require calibration")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "calibration_required", calibration_required)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    name: str
    file: str
    sha256: str
    opset: int
    required: bool
    streaming: bool
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    precision: PrecisionPolicy

    def __post_init__(self) -> None:
        if not self.name or not self.file:
            raise ValueError("artifact name/file must be non-empty")
        if len(self.sha256) != 64:
            raise ValueError("artifact SHA-256 must contain 64 hex characters")
        try:
            int(self.sha256, 16)
        except ValueError as error:
            raise ValueError("artifact SHA-256 must be hexadecimal") from error
        if self.opset <= 0:
            raise ValueError("ONNX opset must be positive")
        input_names = [item.name for item in self.inputs]
        output_names = [item.name for item in self.outputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError("artifact input names must be unique")
        if len(output_names) != len(set(output_names)):
            raise ValueError("artifact output names must be unique")


@dataclass(frozen=True, slots=True)
class BundleManifest:
    voxera_version: str
    weights_status: str
    sample_rate: int
    acoustic_hop_ms: int
    condition_hop_ms: int
    source_checkpoint_sha256: str | None = None
    schema: int = DEPLOYMENT_SCHEMA
    artifacts: tuple[ModelArtifact, ...] = ()

    def __post_init__(self) -> None:
        if self.schema != DEPLOYMENT_SCHEMA:
            raise ValueError(f"unsupported deployment schema: {self.schema}")
        if not self.voxera_version or not self.weights_status:
            raise ValueError("version and weights_status must be non-empty")
        if min(self.sample_rate, self.acoustic_hop_ms, self.condition_hop_ms) <= 0:
            raise ValueError("timebase values must be positive")
        if self.source_checkpoint_sha256 is not None:
            if len(self.source_checkpoint_sha256) != 64:
                raise ValueError("source checkpoint SHA-256 must contain 64 hex characters")
            try:
                int(self.source_checkpoint_sha256, 16)
            except ValueError as error:
                raise ValueError(
                    "source checkpoint SHA-256 must be hexadecimal"
                ) from error

        names = [artifact.name for artifact in self.artifacts]
        files = [artifact.file for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("artifact names must be unique")
        if len(files) != len(set(files)):
            raise ValueError("artifact files must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def load_bundle_manifest(path: Path) -> BundleManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = tuple(
        ModelArtifact(
            name=item["name"],
            file=item["file"],
            sha256=item["sha256"],
            opset=item["opset"],
            required=item["required"],
            streaming=item["streaming"],
            inputs=tuple(TensorSpec(**tensor) for tensor in item["inputs"]),
            outputs=tuple(TensorSpec(**tensor) for tensor in item["outputs"]),
            precision=PrecisionPolicy(**item["precision"]),
        )
        for item in payload.get("artifacts", ())
    )
    return BundleManifest(
        schema=payload["schema"],
        voxera_version=payload["voxera_version"],
        weights_status=payload["weights_status"],
        sample_rate=payload["sample_rate"],
        acoustic_hop_ms=payload["acoustic_hop_ms"],
        condition_hop_ms=payload["condition_hop_ms"],
        source_checkpoint_sha256=payload.get("source_checkpoint_sha256"),
        artifacts=artifacts,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(directory: Path, manifest: BundleManifest) -> None:
    for artifact in manifest.artifacts:
        path = directory / artifact.file
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != artifact.sha256:
            raise ValueError(
                f"SHA-256 mismatch for {artifact.name}: "
                f"expected {artifact.sha256}, got {actual}"
            )
