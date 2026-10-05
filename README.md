# Voxera

Lightweight offline semantic-prosodic neural voice conversion engine for edge devices.

> Status: **M1 / pre-alpha**. The core API is native to Voxera. A real zero-shot WAV-to-WAV reference path is now available through an external Apache-2.0 MeanVC2 baseline while Voxera-native neural components are implemented.

## Goals

- fully offline inference;
- preserve linguistic content while converting speaker identity;
- preserve and later model prosody, emphasis, sentence intent, and language style;
- one model contract across Android, Windows, macOS, and Linux;
- exportable inference path targeting compact native/ONNX runtimes;
- small, inspectable components rather than a single opaque end-to-end dependency.

## Architecture

```text
Audio
  ├─ Content Encoder ──────┐
  ├─ F0 / Pitch Extractor ─┼─> Condition Fusion -> Voice Decoder -> PCM
  ├─ Prosody Encoder* ─────┤
  └─ Semantic Sidecar* ────┘
                + Speaker Embedding
```

`*` planned for later milestones. Semantic analysis will be asynchronous so it never blocks the real-time audio path.

## M0 core

The Voxera reference package already contains:

- dependency-light PCM WAV I/O;
- deterministic reference resampling;
- typed protocols for Content Encoder, Pitch Extractor, and Voice Decoder;
- a model-agnostic `ReferencePipeline`;
- CI tests and a small real-time-factor smoke benchmark.

The reference resampler is deliberately simple and is **not** intended to be the final production resampler.

## M1 executable baseline

M1 uses MeanVC2 through audio.cpp as a temporary quality/latency oracle. Neither project is vendored into Voxera.

Bootstrap the reference backend on a development machine:

```bash
sh tools/bootstrap_reference.sh
```

Verify it:

```bash
voxera baseline doctor
```

Run real zero-shot conversion:

```bash
voxera baseline convert source.wav target.wav converted.wav
```

The source is what is being said; the target WAV is the reference voice. Reference assets are kept below `.cache/reference/` and never committed.

See [docs/M1_REFERENCE_BASELINE.md](docs/M1_REFERENCE_BASELINE.md) for the replacement plan.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest -q
python benchmarks/smoke.py
```

Inspect a 16-bit PCM WAV:

```bash
voxera inspect input.wav
```

## Roadmap

1. **M0 — Core contract:** audio I/O, component interfaces, deterministic tests and benchmark harness. **Done.**
2. **M1 — Offline VC baseline:** real WAV-to-WAV reference conversion plus a reproducible benchmark contract. **In progress.**
3. **M2 — Prosody:** compact prosody/style encoder for pitch contour, energy, pace, pauses and emphasis.
4. **M3 — Semantic sidecar:** offline streaming ASR/language/intent conditioning without blocking audio.
5. **M4 — Edge runtime:** native/ONNX export, FP16/INT8/Q4, reduced runtime and Android/desktop integration.
6. **M5 — Streaming:** chunked inference, cross-fade/state handling, latency and power optimization.

## Licensing

Source code in this repository is licensed under Apache-2.0 unless a file says otherwise. Model weights and datasets can have separate licenses and must be tracked independently. See [`THIRD_PARTY.md`](THIRD_PARTY.md).
