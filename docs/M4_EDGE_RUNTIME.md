# M4 edge runtime deployment contract

M4 starts by making deployment **reproducible and inspectable** before applying
quantization.

No quantization result is claimed until trained weights and calibration data
exist.

## Bundle layout

A production export is generated with:

```bash
sh tools/export_runtime_bundle.sh
```

The no-argument shell entry point automatically selects the newest
`joint-refinement-epoch*.pt` checkpoint from `artifacts/checkpoints/`.

Output:

```text
artifacts/runtime_bundle/
├── bundle.json
├── contentnet.onnx
├── timbrenet.onnx
├── condition_fusion.onnx
├── decodernet.onnx
└── lite_vocoder.onnx
```

The five required M1 neural components are exported as separate graphs. Native
frontend, YIN pitch, cadence selection, iFFT and overlap-add remain runtime DSP
rather than being hidden inside ONNX.

## Production safety

`training.export_bundle` refuses to export without a joint-refinement
checkpoint.

The only bypass is the explicit development flag:

```text
--allow-untrained
```

That mode is used by CI only. Its manifest is marked
`weights_status = "untrained-smoke"` and contains no source checkpoint hash.

A real bundle records the SHA-256 of the source joint checkpoint.

## Manifest schema

`bundle.json` records:

- Voxera version;
- deployment schema version;
- source checkpoint SHA-256;
- sample rate and 10/40 ms timebases;
- every ONNX filename and SHA-256;
- ONNX opset;
- required/optional status;
- streaming/non-streaming status;
- named input and output tensor contracts;
- static and symbolic dimensions;
- precision candidates;
- which precision modes require calibration.

Bundle loading verifies every artifact hash before execution.

## Core graph contracts

### ContentNet

```text
features   [B, T, 80]
cache      [12, B, 64, 192]
        ↓
content    [B, T, 256]
next_cache [12, B, 64, 192]
```

### TimbreNet

```text
features [B, reference_T, 80]
      ↓
speaker  [B, 256]
```

### ConditionFusion

```text
content [B, T40, 256]
pitch   [B, T40, 3]
speaker [B, 256]
      ↓
condition [B, T40, 256]
```

Attention diagnostics are intentionally not exported in the deployment graph.

### DecoderNet

```text
conditions [B, T40, 256]
cache      [8, B, 16, 192]
        ↓
mel        [B, T10, 80]
next_cache [8, B, 16, 192]
```

`T10 = 4 × T40`.

### LiteVocoder

```text
mel           [B, T10, 80]
cache         [8, B, 16, 192]
           ↓
log_magnitude [B, T10, 161]
phase         [B, T10, 161]
next_cache    [8, B, 16, 192]
```

The runtime then performs the native 320-point iFFT and overlap-add.

## Precision policy

The first manifest deliberately uses FP32 as the correctness baseline.

Each core model currently declares these **candidates**, not promises:

- FP16;
- calibrated static INT8.

INT8 is marked calibration-required. Voxera will not produce a production INT8
bundle from random inputs.

Q4 weight-only is not enabled in the first contract. It is backend-specific and
must first prove acceptable quality and operator coverage on the actual target
runtime.

## CI

CI exports the same five ONNX graphs using explicit random initialization,
validates each graph with `onnx.checker`, writes the deployment manifest, then
re-loads the manifest and verifies every SHA-256.

This validates architecture/export compatibility without pretending that the
generated smoke bundle has useful voice-conversion weights.

## Next M4 steps

Once trained checkpoints/calibration speech exist:

1. establish FP32 and FP16 parity tolerances;
2. collect per-model activation calibration ranges;
3. produce static INT8 QDQ candidates;
4. compare CPU / Android NNAPI / QNN HTP operator coverage;
5. measure model size, peak RSS, RTF, power and quality deltas;
6. enable only precision modes that satisfy quality + performance gates.
