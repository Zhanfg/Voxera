# M1 training path

The M1 graph is intentionally small, but it still needs strong supervision.
Voxera therefore uses a reproducible teacher-cache stage before training its
native models.

## Teacher pin

The initial teacher is MeanVC2 at:

```text
repository: https://github.com/ASLP-lab/MeanVC2
commit:     13acf84c1bf135ea5edad9c245b345289b06b33e
```

Do not silently train against whatever MeanVC2 happens to have on `main`.
Changing the teacher changes the target feature distribution and must be an
explicit experiment.

## Official MeanVC2 extraction paths

Voxera calls MeanVC2's own preprocessing scripts instead of copying their
implementation:

- `preprocess/extract_bn_80ms.py`
  - Fast-U2++ teacher;
  - 256-dimensional BN;
  - one BN step every 40 ms.
- `preprocess/extract_spk_emb.py`
  - WavLM-Large + ECAPA-TDNN teacher;
  - global 256-dimensional speaker embedding.
- `preprocess/extract_mel.py`
  - 80-bin mel target;
  - 10 ms frame cadence.

The teacher code/weights remain outside the Voxera runtime.

## Cache layout

```text
data/train_wavs/
  <utterance>.wav

artifacts/teacher_cache/
  bn/
    <utterance>.npy
  mel/
    <utterance>.npy
  speaker/
    <utterance>.npy
  manifest.jsonl
  provenance.json
```

`training/teacher_cache.py` rejects missing IDs, non-finite arrays, incorrect
dimensions, and mismatched cache sets before training starts. The manifest also
stores a SHA-256 hash for each source WAV so stale teacher outputs can be
detected by later tooling.

## One-command training

Once real training WAVs exist, the complete M1 path has one no-argument entry
point:

```bash
sh tools/train_m1.sh
```

It performs, in order:

1. Voxera environment + WAV format audit;
2. pinned MeanVC2 bootstrap;
3. stale-aware teacher cache extraction;
4. ContentNet + TimbreNet training;
5. ConditionFusion + DecoderNet + LiteVocoder training;
6. full-graph joint refinement.

MeanVC2 and Voxera can use different Python interpreters:

```text
VOXERA_TEACHER_PYTHON=/path/to/meanvc2/python
VOXERA_TRAIN_PYTHON=/path/to/voxera/python
```

This is intentional: teacher preprocessing is a research-only dependency stack,
not part of the deployable Voxera runtime.

## Teacher-cache freshness

Before every extraction pass, Voxera compares the current WAV SHA-256 values
against the previous manifest. Changed or removed utterances have their cached
BN, mel, and speaker arrays deleted before MeanVC2 runs. A teacher-commit change
invalidates the complete cache.

The old manifest/provenance files are removed as soon as stale data is detected,
so a failed partial extraction cannot later be mistaken for a valid cache.

## Implemented encoder training

The first two stages are executable:

```bash
sh tools/train_encoders.sh
```

The training loop consumes only `manifest.jsonl`, recomputes Voxera-native
frontend features from the source WAV, and uses the cached MeanVC2 tensors as
supervision.

ContentNet aligns its dense 10 ms representation to the teacher's 40 ms BN
cadence using the common valid prefix. TimbreNet trains from two independent
reference crops against the normalized WavLM+ECAPA embedding.

Encoder checkpoints record the exact teacher commit and SHA-256 of the manifest
used for that run. This prevents an old checkpoint from being mistaken for one
trained against a newer cache.

## Implemented generator training

The acoustic and waveform stages are also executable:

```bash
sh tools/train_generators.sh
```

The first acoustic pass trains `ConditionFusion + DecoderNet` directly from
teacher BN, teacher speaker embedding, native source pitch, and teacher mel.
This removes student-encoder noise from the initial decoder optimization.

LiteVocoder is trained independently from teacher mel aligned to the original
16 kHz source waveform. Each mel frame maps to one 160-sample waveform hop.

All staged trainers now use actual-count gradient averaging for partial
accumulation groups and write provenance-bearing atomic checkpoints.

## Implemented joint refinement

Stage 5 is executable:

```bash
sh tools/train_joint.sh
```

Joint refinement restores the newest staged checkpoints only when their teacher
commit and manifest SHA-256 match the active teacher cache. It then trains the
full five-model graph with four simultaneous constraints:

- ContentNet vs Fast-U2++ BN;
- TimbreNet vs WavLM+ECAPA embedding;
- DecoderNet vs teacher mel;
- LiteVocoder vs aligned source waveform/Fourier targets.

The auxiliary content and speaker losses are intentionally retained during
end-to-end optimization to limit representation collapse.

## Training order

1. **ContentNet**
   - teacher: Fast-U2++ BN;
   - objective: MSE + cosine + temporal delta.
2. **TimbreNet**
   - teacher: WavLM + ECAPA embedding;
   - objective: cosine + normalized MSE + crop consistency.
3. **ConditionFusion + DecoderNet**
   - target: MeanVC2-aligned 80-bin mel;
   - freeze or slowly fine-tune ContentNet/TimbreNet initially.
4. **LiteVocoder**
   - target: ground-truth waveform/Fourier frames;
   - Vocos remains an optional perceptual teacher.
5. **Joint refinement**
   - low learning rate;
   - preserve content/speaker auxiliary losses to prevent entanglement.

Stages 1 through 5 and the one-command orchestrator are implemented. The training-system code path is complete; the remaining M1 gates are producing real teacher caches/checkpoints on a sufficiently large licensed dataset and measuring conversion quality/latency against the MeanVC2 oracle.
