# Voxera

Lightweight offline semantic-prosodic neural voice conversion engine for edge devices.

> Status: **Phase 0 / pre-alpha**. The repository currently defines the reference audio and model contracts. It does not ship a pretrained voice-conversion model yet.

## Goals

- fully offline inference;
- preserve linguistic content while converting speaker identity;
- preserve and later model prosody, emphasis, sentence intent, and language style;
- one model contract across Android, Windows, macOS, and Linux;
- exportable inference path targeting ONNX Runtime;
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

## Phase 0 contract

The current reference package contains:

- dependency-light PCM WAV I/O;
- deterministic reference resampling;
- typed protocols for Content Encoder, Pitch Extractor, and Voice Decoder;
- a model-agnostic `ReferencePipeline`;
- CI tests and a small real-time-factor smoke benchmark.

The reference resampler is deliberately simple and is **not** intended to be the final production resampler.

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

1. **M0 — Core contract:** audio I/O, component interfaces, deterministic tests and benchmark harness.
2. **M1 — Offline VC baseline:** content encoder + F0 extractor + target-speaker decoder, WAV-to-WAV.
3. **M2 — Prosody:** compact prosody/style encoder for pitch contour, energy, pace, pauses and emphasis.
4. **M3 — Semantic sidecar:** offline streaming ASR/language/intent conditioning without blocking audio.
5. **M4 — Edge runtime:** ONNX export, FP16/INT8, reduced-operator runtime and Android/desktop integration.
6. **M5 — Streaming:** chunked inference, cross-fade/state handling, latency and power optimization.

## Licensing

Source code in this repository is licensed under Apache-2.0 unless a file says otherwise. Model weights and datasets can have separate licenses and must be tracked independently. See [`THIRD_PARTY.md`](THIRD_PARTY.md).
