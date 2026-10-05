# Voxera

Lightweight offline semantic-prosodic neural voice conversion engine for edge devices.

> Status: **M1 / pre-alpha**. A real zero-shot WAV-to-WAV reference path is available through MeanVC2. Voxera now also owns its streaming acoustic frontend and the first trainable ContentNet architecture.

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
  ↓
Native 80-bin log-mel frontend
  ↓
ContentNet ───────────────────┐
F0 / Pitch Extractor* ────────┤
Prosody Encoder* ─────────────┼─> Condition Fusion -> Voice Decoder* -> PCM
Semantic Sidecar* ────────────┤
Speaker / Style Encoder* ─────┘
```

`*` not Voxera-native yet.

## Native M1 components

### Acoustic frontend

- 16 kHz audio;
- 25 ms frame / 10 ms hop;
- 80-bin log-mel filterbank;
- explicit pre-emphasis and Povey-style windowing;
- offline and streaming APIs;
- irregular chunk boundaries produce the same features as offline extraction.

### ContentNet

The first Voxera neural component is a compact causal TCN:

```text
80-d log-mel @ 10 ms
        ↓
80 → 192 projection
        ↓
12 causal depthwise TCN blocks
dilation 1,2,4,8,16,32 × 2
        ↓
192 → 256 projection
        ↓
dense content @ 10 ms
        ↓
cadence selector
        ↓
256-d content @ 40 ms
```

Design targets:

- about 1.86M parameters;
- about 7.1 MiB in FP32 before graph/runtime overhead;
- about 1.9 MB raw weight payload if fully INT8-quantized;
- fixed-shape streaming state rather than attention KV cache;
- ONNX-exportable;
- trained by distillation against the MeanVC2/Fast-U2++ bottleneck teacher.

The runtime package does **not** depend on PyTorch. PyTorch and ONNX belong to the optional training toolchain only.

See [training/README.md](training/README.md).

## M1 executable teacher baseline

MeanVC2 through audio.cpp remains the temporary quality/latency oracle while native components are trained and connected.

```bash
sh tools/bootstrap_reference.sh
voxera baseline doctor
voxera baseline convert source.wav target.wav converted.wav
```

Reference assets stay below `.cache/reference/` and are never committed.

See [docs/M1_REFERENCE_BASELINE.md](docs/M1_REFERENCE_BASELINE.md).

## Development

Core/runtime checks:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest -q
python benchmarks/frontend.py
```

ContentNet development:

```bash
pip install -e ".[train]"
python training/smoke_contentnet.py
python training/export_contentnet.py --output artifacts/contentnet.onnx
```

## Roadmap

1. **M0 — Core contract:** audio I/O, component interfaces, deterministic tests and benchmark harness. **Done.**
2. **M1 — Offline VC baseline:** teacher reference + native acoustic frontend + native ContentNet + remaining VC path. **In progress.**
3. **M2 — Prosody:** compact prosody/style encoder for pitch contour, energy, pace, pauses and emphasis.
4. **M3 — Semantic sidecar:** offline streaming ASR/language/intent conditioning without blocking audio.
5. **M4 — Edge runtime:** native/ONNX export, FP16/INT8/Q4, reduced runtime and Android/desktop integration.
6. **M5 — Streaming:** chunked inference, cross-fade/state handling, latency and power optimization.

## Licensing

Source code in this repository is Apache-2.0 unless a file says otherwise. Model weights and datasets can have separate licenses and must be tracked independently. See [`THIRD_PARTY.md`](THIRD_PARTY.md).
