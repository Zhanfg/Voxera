from __future__ import annotations

import tempfile
from pathlib import Path

import onnx

from training.export_bundle import main as export_main
from voxera.deployment import load_bundle_manifest, verify_bundle


def main() -> int:
    import sys

    with tempfile.TemporaryDirectory(prefix="voxera-export-smoke-") as temporary:
        output = Path(temporary) / "bundle"
        previous = sys.argv
        try:
            sys.argv = [
                "export_bundle",
                "--output-dir",
                str(output),
                "--allow-untrained",
                "--chunk-frames",
                "4",
                "--condition-frames",
                "2",
                "--reference-frames",
                "8",
            ]
            export_main()
        finally:
            sys.argv = previous

        manifest = load_bundle_manifest(output / "bundle.json")
        verify_bundle(output, manifest)

        expected = {
            "contentnet",
            "timbrenet",
            "condition_fusion",
            "decodernet",
            "lite_vocoder",
        }
        actual = {artifact.name for artifact in manifest.artifacts}
        if actual != expected:
            raise RuntimeError(f"unexpected exported model set: {actual}")
        if manifest.weights_status != "untrained-smoke":
            raise RuntimeError("smoke export must be explicitly marked untrained")
        if manifest.source_checkpoint_sha256 is not None:
            raise RuntimeError("untrained smoke must not claim checkpoint provenance")

        for artifact in manifest.artifacts:
            onnx.checker.check_model(onnx.load(output / artifact.file))

        print(f"artifacts={len(manifest.artifacts)}")
        print(f"weights_status={manifest.weights_status}")
        print(f"manifest={output / 'bundle.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
