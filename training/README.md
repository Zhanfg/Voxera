# Voxera training

Training dependencies are deliberately separated from the runtime package.

## ContentNet

`CausalContentNet` is the first Voxera-native neural component.

### Contract

```text
80-d log-mel @ 10 ms
        ↓
Linear 80 → 192
        ↓
12 causal depthwise TCN blocks
dilation = 1,2,4,8,16,32 × 2
        ↓
Linear 192 → 256
        ↓
dense 256-d content @ 10 ms
        ↓
runtime cadence selector
        ↓
256-d content @ 40 ms
```

The model has about 1.86M parameters. Its fixed-shape state is designed for ONNX and edge runtimes:

```text
[layer, batch, 64 history frames, 192 hidden]
```

No attention KV cache is required.

### Teacher

The initial teacher target is MeanVC2's Fast-U2++ bottleneck representation. The teacher emits 256-dimensional content features at the VC cadence; Voxera predicts densely and supervises every fourth output frame.

The first objective combines:

- feature MSE;
- cosine embedding agreement;
- first-order temporal-delta agreement.

This is intentionally not speaker reconstruction training. ContentNet should learn a speaker-resistant linguistic representation before it is connected to the timbre path.

### Smoke test

Install the optional training toolchain and run:

```bash
pip install -e ".[train]"
python training/smoke_contentnet.py
```

Expected default size is approximately 1.86M parameters, and streaming/offline outputs must agree within floating-point tolerance.

### ONNX

```bash
python training/export_contentnet.py --output artifacts/contentnet.onnx
```

The exported graph accepts:

- `features [B,T,80]`
- `cache [12,B,64,192]`

and returns:

- `content [B,T,256]`
- `next_cache [12,B,64,192]`

Cadence selection stays outside the neural graph so chunk length can vary without embedding a mutable phase counter inside ONNX.


## TimbreNet

`TimbreNet` replaces the heavy target-reference speaker encoder path.

### Contract

```text
80-d log-mel
    ↓
80 → 160 projection
    ↓
8 temporal depthwise residual blocks
    ↓
attentive statistics pooling
    ↓
weighted mean + std
    ↓
320 → 256 projection
    ↓
L2-normalized speaker embedding
```

The default model has **949,473 parameters** (about **3.62 MiB FP32** before quantization).

### Teacher

The initial teacher target is MeanVC2's WavLM + ECAPA-TDNN global speaker embedding. Training uses:

- cosine agreement with the teacher;
- normalized embedding MSE;
- same-speaker consistency across independently cropped/augmented reference segments.

The speaker encoder is intentionally separated from the later content-conditioned timbre-token projector. This prevents identity extraction and fine-grained pronunciation/timbre retrieval from becoming one large opaque model.

### Smoke test

```bash
python training/smoke_timbrenet.py
```

The smoke test checks output dimensionality and L2 normalization.
