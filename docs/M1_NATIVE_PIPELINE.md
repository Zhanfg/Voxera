# M1 native pipeline

M1 is not complete merely because every neural block can run independently.
The runtime also needs one explicit contract for how those blocks compose.

## Offline reference path

```text
source PCM
  ↓ resample 16 kHz
native 80-bin frontend
  ├─ ContentNet → dense 10 ms content
  └─ YIN pitch → dense 10 ms F0/periodicity
                ↓ align common prefix + 40 ms cadence

target reference PCM
  ↓ native 80-bin frontend
TimbreNet
  ↓ 256-d speaker embedding (computed once / reusable)

content + pitch + speaker
  ↓ ConditionFusion
256-d condition @ 40 ms
  ↓ DecoderNet
80-bin mel @ 10 ms
  ↓ LiteVocoder
log-magnitude + phase @ 10 ms
  ↓ native StreamingISTFT
16 kHz PCM
```

## Backend independence

`NativeOfflinePipeline` does not import PyTorch or ONNX Runtime.

Instead, five small protocols are supplied by a backend:

- `ContentModel`
- `SpeakerModel`
- `ConditionModel`
- `AcousticDecoderModel`
- `VocoderModel`

This allows one orchestration layer to run against:

- Python/PyTorch during training;
- ONNX Runtime on desktop;
- QNN/HTP or another accelerator on Android;
- future native C++ kernels.

## Speaker reuse

Reference-speaker extraction is explicitly separated:

```text
speaker = pipeline.prepare_speaker(reference)
pipeline.convert(source_a, speaker)
pipeline.convert(source_b, speaker)
...
```

A real-time session therefore does not repeatedly run TimbreNet.

## Traceability

Every conversion returns a compact shape trace:

- source samples;
- frontend frames;
- pitch frames;
- 40 ms condition frames;
- 10 ms mel frames;
- Fourier frames;
- final PCM samples.

This is intended for regression tests and mobile performance diagnostics.


## Exact duration contract

Snipped-edge feature extraction naturally loses the final analysis-window tail.
That is unacceptable for conversion because repeated utterances would slowly
drift from the original timeline.

Before source analysis, Voxera now computes the number of 10 ms mel frames
needed to cover the original resampled waveform. It then derives the number of
40 ms condition frames and right-pads **analysis audio only** until the 40 ms
pitch window can produce the final required cadence frame.

For example, a 16,000-sample source requires:

```text
100 output mel frames
→ 25 condition frames
→ dense frame indices 3, 7, ..., 99
→ 16,480 analysis samples after right padding
→ 16,000 generated/output samples
```

The generated waveform is finally cropped to the exact original resampled
sample count. This means duration is preserved for arbitrary input lengths,
including inputs that are not aligned to the 160-sample vocoder hop.

The trace reports both the padded analysis length and pre-crop generated length
so regressions remain visible rather than hidden.
