# M1 target timbre path

The reference MeanVC2 pipeline extracts a 256-dimensional global speaker
embedding using WavLM upstream features plus ECAPA-TDNN. This is effective but
too large for Voxera's final edge runtime.

## TimbreNet

Voxera replaces that target-reference path with a compact fbank encoder:

```text
target reference audio
        ↓
native 80-bin log-mel frontend
        ↓
80 → 160 projection
        ↓
8 depthwise temporal residual blocks
        ↓
attentive statistics pooling
        ↓
mean + std
        ↓
320 → 256 projection
        ↓
L2-normalized speaker embedding
```

The reference encoder is allowed symmetric context because target voice
encoding happens once per reference, not on the real-time source audio path.

## Distillation

The first teacher is MeanVC2's WavLM + ECAPA speaker embedding.

Training should use multiple crops/augmentations from the same reference
speaker and combine:

- teacher cosine agreement;
- normalized embedding MSE;
- same-speaker crop consistency.

This teaches TimbreNet to preserve identity while becoming insensitive to
reference length, recording level, and incidental phonetic content.

## What this does not solve

A single global embedding is not enough for all fine-grained timbre behavior.
MeanVC2 uses its Universal Timbre Token Encoder (UTTE) to turn the global
speaker representation into content-conditioned timbre tokens.

Voxera will add a smaller token projector after TimbreNet rather than enlarging
TimbreNet itself. Keeping these responsibilities separate makes the edge model
easier to quantize and debug.
