from __future__ import annotations

import tempfile
from pathlib import Path

import onnx

from training.export_bundle import export_edge_bundle


EXPECTED_MODELS = {
    "contentnet",
    "timbrenet",
    "condition_fusion",
    "prosodynet",
    "prosody_conditioner",
    "semantic_conditioner",
    "decoder",
    "lite_vocoder",
}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="voxera-edge-export-") as temporary:
        root = Path(temporary)
        output = root / "bundle"
        manifest = export_edge_bundle(
            output,
            checkpoint_dir=root / "no-checkpoints",
            chunk_frames=8,
        )

        if manifest["schema"] != 1:
            raise RuntimeError("unexpected bundle schema")
        if set(manifest["models"]) != EXPECTED_MODELS:
            raise RuntimeError("unexpected model set")
        if manifest["totals"]["unique_parameters"] <= 0:
            raise RuntimeError("parameter total must be positive")

        for name, item in manifest["models"].items():
            path = output / item["file"]
            if not path.is_file() or path.stat().st_size <= 0:
                raise RuntimeError(f"missing ONNX graph for {name}")
            if len(item["sha256"]) != 64:
                raise RuntimeError(f"invalid SHA-256 for {name}")
            onnx.checker.check_model(onnx.load(path))

        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("bundle manifest is missing")

        print(f"models={len(manifest['models'])}")
        print(f"unique_parameters={manifest['totals']['unique_parameters']}")
        print(f"onnx_bytes={manifest['totals']['onnx_bytes']}")
        print(f"manifest={manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
