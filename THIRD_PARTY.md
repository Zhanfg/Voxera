# Third-party policy

Voxera's core source code is Apache-2.0. Third-party code and model weights must be reviewed independently before they are vendored or redistributed.

## Current runtime dependencies

- NumPy — BSD-3-Clause.

## M1 reference-only dependencies

These components are **not vendored** into Voxera and are not required by the core package:

- **audio.cpp** — Apache-2.0. Used as the executable native reference runtime for M1.
- **MeanVC2** — Apache-2.0. Used as the M1 zero-shot voice-conversion teacher/reference model. Teacher feature extraction is pinned to commit `13acf84c1bf135ea5edad9c245b345289b06b33e`; source and checkpoints remain outside the Voxera runtime.
- **Vocos** — MIT. Used as the vocoder quality/teacher reference. Vocos source code and pretrained weights are not vendored into the Voxera runtime.

Reference assets live below `.cache/reference/` and teacher assets below `.cache/teacher/`; both are excluded from Git. Runtime reference provenance is recorded by `tools/bootstrap_reference.sh`, while the training teacher is pinned by `tools/bootstrap_teacher.sh`.

No pretrained third-party model weights are committed to this repository.

## M3 ASR adapter targets

ASR source trees are **not vendored** into Voxera and are not runtime
dependencies of the Python core:

- **sherpa-onnx v1.13.8** — Apache-2.0. Primary offline/streaming ASR adapter
  target, including the implemented optional Python reference adapter plus
  planned Android JNI and Qualcomm QNN/HTP deployment paths.
- **whisper.cpp v1.9.4** — MIT. Secondary compatibility and ASR quality
  baseline.

`tools/bootstrap_asr.sh` pins these source versions below `.cache/asr/`.
It does not download ASR model weights.

ASR model weights must be reviewed independently: model licenses can differ
from the framework source license.

## Integration rule

Every future model or library integration must record:

1. upstream project and exact version or commit;
2. source-code license;
3. model/data license when weights are involved;
4. whether redistribution is permitted;
5. whether attribution or NOTICE text is required.

GPL components must not be copied into the Apache-2.0 core. If a GPL tool is useful for research or data preparation, keep it outside the distributed core and document the boundary explicitly.
