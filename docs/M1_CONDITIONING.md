# M1 condition fusion

M1 now has four distinct responsibilities:

1. acoustic frontend — convert PCM to stable 10 ms fbank frames;
2. ContentNet — extract linguistic/content representations;
3. Pitch — preserve source F0, voicing, and periodicity;
4. TimbreNet — identify the target speaker from reference audio.

Condition Fusion combines those streams without collapsing them into one large
encoder.

## Time alignment

The fbank frontend uses a 25 ms analysis window while native pitch uses 40 ms.
Both use a 10 ms hop, but a finite utterance therefore produces different tail
frame counts.

Voxera aligns by **frame start time**:

```text
common = min(content_frames, pitch_frames)
crop both streams to common
apply the same 4-frame cadence: 3, 7, 11, ...
```

The resulting decoder cadence is 40 ms.

## Pitch representation

Each decoder frame receives three pitch features:

1. `log2(F0 / 55 Hz)` for voiced frames, otherwise zero;
2. voiced flag;
3. periodicity/confidence.

This keeps absolute pitch information while making the scale easier for a
small neural network to model.

## Timbre tokens

A single 256-d TimbreNet embedding says *who* the target is, but the decoder
also needs content-dependent timbre cues.

The compact token projector expands the global embedding into eight 128-d
learned tokens. ContentNet features query those tokens with scaled dot-product
attention:

```text
speaker embedding
      ↓
8 learned timbre tokens ─── keys / values
                           ↑
content 256 → query 128 ───┘
      ↓
content-conditioned timbre 128
```

This follows the same conceptual separation as MeanVC2's global speaker
embedding plus Universal Timbre Token Encoder, but keeps the Voxera module
small and explicit.

## Fused decoder condition

Per 40 ms frame:

```text
content 256
+ timbre context 128
+ pitch hidden 64
       ↓
gated residual fusion
       ↓
decoder condition 256
```

The fusion model is independent from the eventual acoustic decoder, so decoder
architectures can be replaced without retraining the content/pitch APIs.
