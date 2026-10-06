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


## Condition Fusion

`ConditionFusion` combines ContentNet, pitch conditions, and TimbreNet into the
per-frame decoder condition.

### Contract

```text
content 256 @ 40 ms ────────────────┐
speaker 256 → 8×128 timbre tokens ──┼─> content-conditioned attention
pitch 3 → 64 hidden ────────────────┘
                         ↓
                 gated residual fusion
                         ↓
                 decoder condition 256
```

The default fusion model has **630,720 parameters** (about **2.41 MiB FP32**).

The timbre-token projector is deliberately separate from TimbreNet. TimbreNet
answers *who is speaking*; token attention answers *which target-timbre cues are
relevant for this content frame*.

### Smoke test

```bash
python training/smoke_condition_fusion.py
```

The smoke test validates output shape and confirms each frame's timbre
attention weights sum to one.


## DecoderNet

`DecoderNet` maps the 40 ms fused condition stream to 10 ms log-mel frames.

### Contract

```text
condition 256 @ 40 ms
       ↓
linear 256 → 4×192
       ↓ reshape
hidden 192 @ 10 ms
       ↓
8 causal depthwise TCN blocks
       ↓
80-bin log-mel @ 10 ms
```

The default decoder has **1,406,672 parameters** (about **5.37 MiB FP32**).

The first objective combines frame-level mel L1 with first- and second-order
temporal reconstruction losses. This discourages over-smoothed trajectories
while keeping the decoder compact and deterministic.

### Smoke test

```bash
python training/smoke_decoder.py
```

The smoke test verifies exact offline/streaming agreement across irregular
condition chunks.


## LiteVocoder

`LiteVocoder` maps 10 ms mel frames to Fourier synthesis parameters.

### Contract

```text
80-bin mel @ 10 ms
       ↓
80 → 192 projection
       ↓
8 causal depthwise TCN blocks
       ↓
192 → 322 spectral head
       ↓
161 log-magnitude + 161 phase
       ↓
native 320-point iFFT + 50% overlap-add
       ↓
160 PCM samples per frame
```

The default neural generator has **1,271,554 parameters** (about **4.85 MiB FP32**).

The FFT is intentionally outside the neural graph. This keeps ONNX export
limited to ordinary linear/normalization/convolution operations while the
runtime owns the deterministic overlap state.

### Initial loss

The first training objective combines:

- log-magnitude L1;
- magnitude-weighted circular phase distance;
- waveform L1 after differentiable Fourier reconstruction.

A later quality stage may add multi-resolution STFT and adversarial feature
matching without changing the deployed generator topology.

### Smoke and export

```bash
python training/smoke_vocoder.py
python training/export_lite_vocoder.py --output artifacts/lite_vocoder.onnx
```


## Encoder training

ContentNet and TimbreNet now have real manifest-driven training loops.

Run both with the no-argument entry point:

```bash
sh tools/train_encoders.sh
```

Defaults:

- manifest: `artifacts/teacher_cache/manifest.jsonl`;
- checkpoints: `artifacts/checkpoints/`;
- 10 epochs;
- automatic CUDA/CPU selection;
- gradient accumulation: 4.

The shell entry point accepts configuration through environment variables rather
than positional arguments:

```text
VOXERA_TEACHER_MANIFEST
VOXERA_CHECKPOINT_DIR
VOXERA_ENCODER_EPOCHS
VOXERA_GRAD_ACCUM
```

### ContentNet stage

For each utterance:

1. read the original source WAV from the validated manifest;
2. compute Voxera's native 80-bin frontend features;
3. run dense 10 ms ContentNet;
4. align the `3,7,11,...` 40 ms student cadence to the common prefix of the
   MeanVC2 Fast-U2++ BN target;
5. optimize MSE + cosine + temporal-delta distillation loss.

Using the native frontend during training is deliberate: the student learns to
match the teacher representation from the same features it will receive on the
device.

### TimbreNet stage

For each utterance, two independently selected reference crops are encoded.
Both are trained against the normalized MeanVC2 WavLM+ECAPA embedding, while a
same-speaker crop-consistency term discourages phonetic leakage.

### Checkpoints

Each checkpoint contains:

- model and optimizer state;
- model configuration;
- training configuration;
- epoch and global step;
- pinned MeanVC2 teacher commit;
- teacher-manifest absolute path and SHA-256.

Checkpoint writes use a temporary file followed by an atomic replace.

CI executes a real one-record, one-epoch CPU training pass for both encoders,
including backward propagation, optimizer update, and checkpoint metadata
validation.


## Generator training

The initial acoustic and waveform stages are executable with:

```bash
sh tools/train_generators.sh
```

The no-argument shell entry point defaults to the validated teacher manifest and
writes checkpoints under `artifacts/checkpoints/`. Configuration can be
overridden with:

```text
VOXERA_TEACHER_MANIFEST
VOXERA_CHECKPOINT_DIR
VOXERA_GENERATOR_EPOCHS
VOXERA_GRAD_ACCUM
```

### ConditionFusion + DecoderNet

This stage intentionally starts from teacher inputs instead of untrained student
encoders:

```text
MeanVC2 BN @ 40 ms ──────────────┐
native source pitch @ 40 ms ─────┼─> ConditionFusion → DecoderNet
teacher speaker embedding ────────┘                    ↓
                                               teacher mel @ 10 ms
```

For each utterance, the trainer takes the common valid prefix across BN, pitch,
and mel, then randomly crops a bounded condition span. The decoder loss combines
mel L1, first-order delta, and second-order acceleration terms.

This isolates acoustic-generation learning from ContentNet/TimbreNet errors.
After the staged models converge, joint refinement can replace teacher
conditions with student conditions.

### LiteVocoder

The initial vocoder stage consumes MeanVC2's aligned 80-bin mel and the original
16 kHz waveform. Mel frames and waveform hops are cropped from the same start
index, preserving exact 160-sample alignment.

The objective remains:

- log-magnitude L1;
- magnitude-weighted circular phase distance;
- waveform L1 after differentiable Fourier reconstruction.

### Gradient accumulation

Encoder and generator trainers average gradients by the **actual** number of
records accumulated before each optimizer step. This matters for the final
partial group when the dataset size is not divisible by the configured
accumulation factor.

### Generator checkpoints

The acoustic checkpoint contains both `condition_fusion` and `decoder`
states. The vocoder checkpoint contains `lite_vocoder`. Both include optimizer
state, configs, epoch/global step, pinned MeanVC2 commit, and manifest SHA-256.

CI executes a real one-record CPU training pass for both generator stages,
including backward propagation and checkpoint validation.


## Joint refinement

After staged checkpoints exist, run the full native graph with:

```bash
sh tools/train_joint.sh
```

The command automatically selects the newest epoch checkpoint for ContentNet,
TimbreNet, the acoustic generator, and LiteVocoder from
`artifacts/checkpoints/`.

Before loading any weights, joint training verifies that every staged checkpoint:

- uses checkpoint schema 1;
- has the expected component type;
- records the pinned MeanVC2 teacher commit;
- was trained against the exact current teacher manifest SHA-256.

The full graph is then optimized as:

```text
native fbank → ContentNet ───────────────────────────────┐
native fbank → TimbreNet ────────────────┐              │
native pitch ─────────────────────────────┼→ Fusion → Decoder → LiteVocoder
                                          │              │          │
teacher speaker ─ timbre auxiliary loss ──┘              │          │
teacher BN ───── content auxiliary loss ─────────────────┘          │
teacher mel ──── mel reconstruction loss ───────────────────────────┤
source PCM ───── waveform/Fourier reconstruction loss ──────────────┘
```

Default joint learning rate is intentionally lower than the staged rates.
Content and timbre auxiliary losses remain active so end-to-end waveform
optimization cannot freely collapse the disentangled representations.

Joint checkpoints contain all five model states, optimizer state, model/train
configuration, manifest provenance, and SHA-256 hashes of every staged
checkpoint used to initialize the run.

CI constructs staged checkpoints with the real trainers and then executes one
full joint backward/optimizer/checkpoint pass, validating the entire checkpoint
chain end to end.


## ProsodyNet bootstrap

M2 introduces a teacher-free acoustic-style bootstrap stage:

```bash
sh tools/train_prosody.sh
```

It requires only ordinary WAV files. Voxera derives local/global pseudo-targets
from pitch, periodicity, energy, pauses, final contour, and emphasis, then trains
the compact ProsodyNet to reconstruct those descriptors while learning richer
32-d local and 16-d phrase-level embeddings.

The local ProsodyNet path is causal and must reproduce offline output exactly
when evaluated in arbitrary chunks. The phrase-level embedding uses whole-phrase
mean/std pooling and is intentionally offline until M5 adds streaming style
state.

The M2 conditioner is kept outside the stable M1 ConditionFusion block. Its
output projection is initialized to zero, so an untrained conditioner has an
exact identity effect on M1 decoder conditions.

ProsodyNet checkpoints include a SHA-256 fingerprint of the WAV collection used
for bootstrap training.


## Semantic conditioning

M3 keeps speech recognition out of the neural audio graph. The trainable piece
inside the graph is a compact `SemanticConditioner`:

```text
semantic snapshot 16 + confidence 1
              ↓
           17 → 64
              ↓
      zero-safe gated residual
              ↓
decoder condition 256
```

The final residual projection is initialized to zero, making a fresh semantic
adapter an exact identity transform. This allows M3 runtime/API integration to
land before semantic-to-acoustic supervision exists.

Current CI validates the identity property. Future real-data training can attach
semantic features to mel/waveform reconstruction while retaining M1 content and
M2 prosody auxiliary constraints.
