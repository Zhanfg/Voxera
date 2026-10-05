# Voxera

Lightweight offline semantic-prosodic neural voice conversion engine for edge devices.

> Status: **M1 / pre-alpha**. A real zero-shot WAV-to-WAV reference path is available through MeanVC2. Voxera now has its own streaming acoustic frontend, compact ContentNet, native pitch path, and compact target-speaker TimbreNet.

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
  ├─ ContentNet ───────────────┤
  ├─ Native F0 / Pitch ─────────┼─> Native Condition Fusion -> DecoderNet -> LiteVocoder -> PCM
  ├─ Prosody Encoder* ─────────┤
  └─ Semantic Sidecar* ────────┘
                    + TimbreNet Speaker Embedding
```

The full M1 generation path is now represented by Voxera-native components. Prosody and semantic conditioning remain later milestones and will stay off the real-time critical path where possible.

## M1 native progress

### Acoustic frontend

- 16 kHz input;
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

### ContentNet

The first Voxera-native neural content encoder is intentionally small and causal:

```text
80-d fbank @ 10 ms
    ↓
80 → 192 projection
    ↓
12 causal depthwise TCN blocks
dilation 1,2,4,8,16,32 × 2
    ↓
192 → 256 projection
    ↓
256-d dense content @ 10 ms
    ↓
chunk-safe cadence selector
    ↓
256-d VC content @ 40 ms
```

Default model size is about **1.86M parameters / 7.08 MiB FP32** before quantization. It uses a fixed bounded convolution cache rather than attention KV caches. The intended first training stage is distillation from the MeanVC2 Fast-U2++ bottleneck representation.

Training dependencies are optional:

```bash
pip install -e ".[train]"
python training/smoke_contentnet.py
python training/export_contentnet.py --output artifacts/contentnet.onnx
```

See [training/README.md](training/README.md) for the model and distillation contract.

Benchmark the native frontend with:

```bash
python benchmarks/frontend.py
```

### Native pitch

Voxera now includes a streaming YIN-style pitch extractor as the low-power/default fallback:

- 40 ms analysis frame / 10 ms hop;
- 50–550 Hz speech range by default;
- F0 + periodicity + voiced/unvoiced output;
- parabolic lag refinement;
- exact offline/streaming parity across irregular chunks;
- no neural/runtime dependency.

A later compact PitchNet can refine this track in noisy or difficult speech while keeping the same public `PitchTrack` contract. See [docs/M1_PITCH.md](docs/M1_PITCH.md).

```bash
python benchmarks/pitch.py
```

### TimbreNet

The target-reference speaker path now has a Voxera-native compact model design:

- 80-bin log-mel input;
- 160 hidden channels;
- 8 depthwise temporal residual blocks;
- attentive statistics pooling;
- 256-d L2-normalized speaker embedding;
- **949,473 parameters / about 3.62 MiB FP32**.

The first training stage distills MeanVC2's WavLM + ECAPA 256-dimensional speaker embedding, with multi-crop consistency to reduce phonetic/reference-length leakage. See [docs/M1_TIMBRE.md](docs/M1_TIMBRE.md).

```bash
python training/smoke_timbrenet.py
```

### Condition Fusion

ContentNet, Pitch and TimbreNet now meet in a compact content-conditioned fusion layer:

- common-prefix time alignment before 40 ms cadence selection;
- pitch encoded as log-F0 + voiced flag + periodicity;
- 8 learned timbre tokens derived from the global speaker embedding;
- content-to-timbre scaled dot-product attention;
- gated residual fusion back to a 256-d decoder condition;
- **630,720 parameters / about 2.41 MiB FP32**.

See [docs/M1_CONDITIONING.md](docs/M1_CONDITIONING.md).

```bash
python training/smoke_condition_fusion.py
```

### DecoderNet

The fused 40 ms condition stream now has a native causal acoustic decoder:

- 256-d condition input;
- direct learned 4× expansion from 40 ms to 10 ms;
- hidden width 192;
- 8 causal depthwise temporal residual blocks;
- 80-bin log-mel output;
- fixed streaming cache;
- **1,406,672 parameters / about 5.37 MiB FP32**.

The direct frame expansion avoids transposed-convolution overlap state, so streaming chunks can exactly reproduce offline inference. See [docs/M1_DECODER.md](docs/M1_DECODER.md).

```bash
python training/smoke_decoder.py
```

### LiteVocoder

The final M1 waveform stage is a compact causal Fourier vocoder:

- 80-bin mel input at 10 ms;
- hidden width 192;
- 8 causal depthwise temporal blocks;
- predicts 161 log-magnitude + 161 phase values per frame;
- native 320-point iFFT with 160-sample hop;
- explicit fixed overlap-add state;
- **1,271,554 parameters / about 4.85 MiB FP32**.

Unlike a time-domain transposed-convolution vocoder, the neural graph stops at spectral parameters. FFT and overlap-add stay in deterministic runtime DSP, which simplifies ONNX/QNN deployment and makes chunk boundaries inspectable.

The current runtime target transform can reconstruct its own spectral targets with sub-micro-scale numerical error and exactly preserves chunked-vs-single-call output. See [docs/M1_VOCODER.md](docs/M1_VOCODER.md).

```bash
python training/smoke_vocoder.py
python training/export_lite_vocoder.py --output artifacts/lite_vocoder.onnx
```

### NativePipeline

The complete M1 component graph now has one backend-independent orchestration contract:

```text
source PCM → frontend → ContentNet + Pitch
target PCM → frontend → TimbreNet (once)
                ↓
          ConditionFusion
                ↓
             DecoderNet
                ↓
            LiteVocoder
                ↓
       native StreamingISTFT
                ↓
             16 kHz PCM
```

`NativeOfflinePipeline` itself imports neither PyTorch nor ONNX Runtime. Model execution is supplied through five small protocols, so the same orchestration can later use PyTorch, ONNX Runtime, QNN/HTP, or native kernels. Target speaker embeddings are prepared once and reusable across multiple source utterances.

The full neural smoke now traverses every trainable M1 module and emits finite PCM through the native iSTFT stage. See [docs/M1_NATIVE_PIPELINE.md](docs/M1_NATIVE_PIPELINE.md).

```bash
python training/smoke_native_pipeline.py
```

### Teacher cache and training targets

The native graph is now paired with a reproducible teacher-extraction contract.
MeanVC2 is pinned to commit `13acf84c1bf135ea5edad9c245b345289b06b33e`
and its official preprocessing scripts produce:

- Fast-U2++ BN: 256-d @ 40 ms;
- WavLM + ECAPA speaker embedding: 256-d;
- mel target: 80 bins @ 10 ms.

Voxera validates all outputs, hashes every source WAV, and records teacher
provenance before any training job can consume the cache.

```bash
sh tools/bootstrap_teacher.sh
sh tools/prepare_teacher_cache.sh
sh tools/train_encoders.sh
```

ContentNet and TimbreNet now have real manifest-driven trainers with gradient
accumulation, clipping, reproducible shuffling, atomic checkpoints, and embedded
teacher/manifest provenance. CI runs one real CPU training step for both.

See [docs/M1_TRAINING.md](docs/M1_TRAINING.md).

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

## Roadmap

1. **M0 — Core contract:** audio I/O, component interfaces, deterministic tests and benchmark harness. **Done.**
2. **M1 — Offline VC architecture:** native end-to-end component graph plus MeanVC2 quality oracle. **Architecture/orchestration complete; teacher cache and encoder training implemented; decoder/vocoder training and quality validation in progress.**
3. **M2 — Prosody:** compact prosody/style encoder for pitch contour, energy, pace, pauses and emphasis.
4. **M3 — Semantic sidecar:** offline streaming ASR/language/intent conditioning without blocking audio.
5. **M4 — Edge runtime:** native/ONNX export, FP16/INT8/Q4, reduced runtime and Android/desktop integration.
6. **M5 — Streaming:** chunked inference, cross-fade/state handling, latency and power optimization.

## Licensing

Source code in this repository is licensed under Apache-2.0 unless a file says otherwise. Model weights and datasets can have separate licenses and must be tracked independently. See [`THIRD_PARTY.md`](THIRD_PARTY.md).
