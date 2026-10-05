# M1 pitch path

Voxera uses a two-tier pitch strategy.

## Tier 1: native low-power baseline

`YinPitchExtractor` is the always-available fallback:

- 16 kHz mono input;
- 40 ms analysis frame;
- 10 ms hop;
- configurable 50–550 Hz speech range;
- YIN cumulative-mean normalized difference;
- parabolic lag refinement;
- periodicity/confidence output;
- deterministic offline and streaming parity;
- NumPy-only reference implementation.

This path requires no neural runtime and is suitable for low-power or degraded modes.

The analysis intentionally does **not** apply a Hann window before the YIN difference function. Testing showed that doing so biased low F0 estimates upward and suppressed periodicity, particularly around 80–110 Hz.

## Tier 2: neural refinement

The planned PitchNet is not required for basic conversion. It will be trained/distilled against a stronger teacher such as FCPE and will share the same `PitchTrack` contract.

The neural model should improve:

- noisy speech;
- breathy/creaky voice;
- rapid transitions;
- octave-error rejection;
- voiced/unvoiced boundaries.

The runtime can then select:

```text
low-power:  YIN only
normal:     PitchNet
robust:     PitchNet + YIN agreement/fallback
```

This keeps the full system neural where neural inference adds value without spending model capacity on a task that has a very cheap deterministic baseline.

## Validation

The unit tests currently require:

- 80, 110, 220 and 440 Hz synthetic tones to be recovered accurately;
- silence to remain unvoiced;
- arbitrary streaming chunk boundaries to match offline extraction;
- invalid frequency/timebase configurations to fail early.

Run the reference benchmark with:

```bash
python benchmarks/pitch.py
```
