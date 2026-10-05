# M1 reference baseline

M1 needs a real WAV-to-WAV quality oracle before Voxera replaces individual neural components.

## Why MeanVC2

The current reference teacher is **MeanVC2**:

- zero-shot speech voice conversion from source speech + target reference;
- Apache-2.0 source/model licensing;
- approximately 18M parameters in the VC model itself;
- streaming-oriented 40 ms and 120 ms variants;
- a recognition/synthesis design that is close to Voxera's intended separation of content and speaker information.

MeanVC2 is a **teacher and benchmark**, not the final Voxera runtime.

Its target-speaker path still relies on a large WavLM-based speaker encoder, so the complete reference system is much larger than Voxera's edge target. M2/M4 will replace that path with a compact speaker/style representation.

## Reference runtime

For reproducibility, M1 uses the Apache-2.0 audio.cpp implementation of MeanVC2 rather than copying the teacher implementation into Voxera.

The reference runtime remains outside the repository under:

```text
.cache/reference/
├── audio.cpp/
├── audio.cpp.commit
├── models/
└── paths.env
```

Bootstrap it on a development machine:

```bash
sh tools/bootstrap_reference.sh
```

Then verify:

```bash
voxera baseline doctor
```

Convert:

```bash
voxera baseline convert source.wav target.wav converted.wav
```

The first bootstrap downloads/builds large reference assets. Those files are intentionally excluded from Git.

## What M1 measures

Every reference conversion should eventually record:

- wall time and real-time factor;
- source/output duration;
- content intelligibility;
- target-speaker similarity;
- peak memory;
- device/backend;
- reference runtime commit and model package.

The reference output becomes a regression target for Voxera-native implementations.

## Replacement sequence

The dependency will be removed incrementally:

1. reproduce the MeanVC2 request/result contract inside Voxera;
2. replace reference audio preprocessing;
3. replace content extraction with Voxera ContentNet;
4. replace the WavLM-based target speaker path with a compact speaker/style encoder;
5. port or distill the 18M conversion decoder;
6. replace the vocoder with the edge vocoder;
7. remove the external reference runtime from normal inference.

At the end of this sequence, audio.cpp/MeanVC2 remains useful only for regression comparison.
