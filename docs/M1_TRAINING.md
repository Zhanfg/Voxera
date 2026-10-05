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

## Bootstrap

The repository provides no-argument shell entry points:

```bash
sh tools/bootstrap_teacher.sh
sh tools/prepare_teacher_cache.sh
```

The bootstrap pins the upstream source. Teacher preprocessing dependencies are
kept outside Voxera's runtime environment because MeanVC2's WavLM/ECAPA stack
is intentionally heavy and is needed only while producing training targets.

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

Training code should consume only the validated Voxera manifest, never raw
directory assumptions.
