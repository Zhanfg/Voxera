# Voxera

Lightweight offline semantic-prosodic neural voice conversion engine for edge devices.

> Status: **M1 / pre-alpha**. A real zero-shot WAV-to-WAV reference path is available through MeanVC2, while Voxera-native streaming components are being implemented underneath the same architecture.

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
  ├─ Native Acoustic Frontend ─┐
  ├─ Content Encoder ──────────┤
  ├─ F0 / Pitch Extractor ─────┼─> Condition Fusion -> Voice Decoder -> PCM
  ├─ Prosody Encoder* ─────────┤
  └─ Semantic Sidecar* ────────┘
                    + Speaker Embedding
```

`*` planned for later milestones. Semantic analysis will be asynchronous so it never blocks the real-time audio path.

## M1 native progress

Voxera now contains its first native streaming speech component:

- 16 kHz acoustic frontend;
- 25 ms frame / 10 ms hop;
- 80-bin log-mel filterbank;
- explicit pre-emphasis and Povey-style windowing;
- offline and streaming APIs;
- chunk-boundary invariant output;
- NumPy-only reference implementation designed for later C++/SIMD porting.

The important invariant is:

```text
offline(audio) == concat(stream(chunk_1), stream(chunk_2), ...)
```

within floating-point tolerance, regardless of irregular input chunk sizes.

Benchmark it with:

```bash
python benchmarks/frontend.py
```

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
python benchmarks/frontend.py
```

Inspect a 16-bit PCM WAV:

```bash
voxera inspect input.wav
```

## Roadmap

1. **M0 — Core contract:** audio I/O, component interfaces, deterministic tests and benchmark harness. **Done.**
2. **M1 — Offline VC baseline:** real WAV-to-WAV reference conversion plus Voxera-native acoustic/content plumbing. **In progress.**
3. **M2 — Prosody:** compact prosody/style encoder for pitch contour, energy, pace, pauses and emphasis.
4. **M3 — Semantic sidecar:** offline streaming ASR/language/intent conditioning without blocking audio.
5. **M4 — Edge runtime:** native/ONNX export, FP16/INT8/Q4, reduced runtime and Android/desktop integration.
6. **M5 — Streaming:** chunked inference, cross-fade/state handling, latency and power optimization.

## Licensing

Source code in this repository is licensed under Apache-2.0 unless a file says otherwise. Model weights and datasets can have separate licenses and must be tracked independently. See [`THIRD_PARTY.md`](THIRD_PARTY.md).
