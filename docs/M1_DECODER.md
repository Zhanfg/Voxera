# M1 acoustic decoder

The acoustic decoder maps Voxera's 40 ms fused condition stream to a 10 ms
80-bin log-mel stream suitable for a neural vocoder.

## Why direct frame expansion

A common upsampling choice is transposed convolution. That introduces overlap
state across chunk boundaries and makes exact streaming equivalence harder to
reason about.

Voxera instead expands each 40 ms condition frame independently:

```text
condition 256
    ↓ linear
4 × hidden-192 subframes
    ↓ reshape
10 ms hidden stream
    ↓ causal temporal refinement
80-bin log-mel
```

This gives deterministic 4× timing before the temporal network.

## DecoderNet

Default architecture:

- input: 256-d fused conditions at 40 ms;
- direct learned 4× frame expansion;
- hidden width: 192;
- 8 causal depthwise temporal residual blocks;
- dilation pattern: 1,2,4,8 × 2;
- output: 80-d log-mel at 10 ms;
- fixed-shape streaming cache.

The acoustic decoder is trained against target mel spectrograms with a
multi-term reconstruction objective:

- frame L1;
- first-order temporal delta L1;
- second-order acceleration L1.

The delta terms discourage over-smoothed speech dynamics without requiring a
large adversarial decoder.

## Boundary invariant

For any sequence of 40 ms conditions:

```text
decoder(all_conditions)
==
concat(decoder(chunk_1), decoder(chunk_2), ...)
```

within floating-point tolerance when the streaming cache is carried forward.

This is required before the model is connected to the real-time audio path.
