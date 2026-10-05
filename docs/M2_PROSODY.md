# M2 prosody core

M2 adds an explicit delivery/prosody path instead of asking the acoustic decoder
to infer sentence style implicitly from pitch alone.

The design has two layers:

1. a deterministic native descriptor that works immediately with no weights;
2. a compact ProsodyNet that learns a richer latent representation while still
   reconstructing the interpretable descriptor.

## Native descriptor

`NativeProsodyExtractor` operates on source PCM plus the existing
`PitchTrack` and emits local features on the 40 ms VC cadence.

Local channels:

```text
0 robust-normalized log-F0
1 local F0 delta
2 periodicity
3 voiced flag
4 robust-normalized log-energy
5 local energy delta
6 pause likelihood
7 emphasis likelihood
```

The utterance-level descriptor contains:

```text
0 median pitch level relative to 110 Hz
1 voiced pitch range
2 loudness proxy
3 energy dynamics
4 voiced ratio
5 pause ratio
6 final pitch contour
7 emphasis ratio
```

A coarse acoustic delivery classifier maps the global descriptor to:

- `neutral`
- `rising`
- `emphatic`
- `animated`
- `subdued`

These are delivery labels, not semantic emotion labels. Acoustic evidence alone
cannot reliably infer semantic valence such as happy vs sarcastic. M3's semantic
sidecar can later provide that information.

## ProsodyNet

The learned encoder remains small and edge-oriented:

```text
80-d native fbank @ 10 ms
+ 3-d pitch features
        ↓
91 → 128 projection
        ↓
8 causal depthwise TCN blocks
dilation 1,2,4,8 × 2
        ↓
local 32-d prosody embedding @ 10 ms
        ↓
runtime 40 ms cadence selection

hidden mean + std over phrase
        ↓
global 16-d style embedding
```

Two auxiliary heads reconstruct the deterministic 8-d local and global
descriptors. This lets ProsodyNet bootstrap from ordinary WAV files without a
large external teacher.

The local path is streaming-safe. The global style vector is phrase-level and
remains offline until M5 introduces explicit streaming style state.

## Zero-safe M1 integration

M2 does not modify the stable M1 ConditionFusion weights.

`ProsodyConditioner` is a separate residual adapter:

```text
M1 fused condition 256 ─────────────────────┐
local prosody 32 + global style 16 → MLP ───┼→ gated residual → 256
                                            ↓
                                      DecoderNet
```

The residual projection is initialized to exactly zero, so a fresh M2 adapter
reproduces M1 output bit-for-bit. Prosody can therefore be trained and rolled
out without invalidating the M1 graph.

## Teacher-free bootstrap

ProsodyNet can be trained from any ordinary 16-bit PCM WAV collection:

```bash
sh tools/train_prosody.sh
```

The initial objective combines:

- local descriptor Smooth-L1;
- global descriptor Smooth-L1;
- a small local-embedding smoothness penalty;
- a small global-embedding norm regularizer.

This is only bootstrap supervision. Once real expressive speech data is
available, later M2 training can add contrastive style learning, augmentation
consistency, emotion/style labels where licensed, and semantic conditioning.

## CI guarantees

CI verifies:

- native descriptor shape and finite-value contracts;
- rising contour, silence/pause, and emphasis behavior;
- exact ProsodyNet local streaming parity;
- zero-safe ProsodyConditioner identity;
- a real CPU forward/backward/optimizer/checkpoint ProsodyNet training pass.
