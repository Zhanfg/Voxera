# M1 LiteVocoder

LiteVocoder is Voxera's final M1 waveform-generation component.

## Reference teacher

MeanVC2 currently uses Vocos for mel-to-waveform synthesis. Vocos is a
Fourier-based single-forward vocoder: a neural backbone predicts spectral
coefficients and an inverse Fourier transform reconstructs audio.

Voxera keeps that useful separation but does not ship the full 13.5M
`vocos-mel-24khz` architecture.

## LiteVocoder architecture

```text
80-bin mel @ 10 ms
      ↓
80 → 192 projection
      ↓
8 causal depthwise TCN blocks
      ↓
192 → 322 spectral head
      ↓
161 log-magnitude + 161 phase values
      ↓
native iFFT + explicit overlap-add state
      ↓
16 kHz PCM
```

The neural graph never contains an FFT. It only predicts Fourier parameters.
This keeps ONNX/mobile export simple and leaves deterministic signal processing
in the native runtime.

## Fourier geometry

Default synthesis geometry:

- sample rate: 16 kHz;
- hop: 160 samples = 10 ms;
- FFT: 320 samples = 20 ms;
- overlap: 50%;
- synthesis window: half-sample sine window.

The sine window satisfies the 50%-overlap squared-window complement property.
Training targets are created by right-padding one overlap region, applying the
same sine window, and taking a real FFT.

Each predicted spectral frame therefore emits exactly one 160-sample hop.
Arbitrary input chunking produces identical PCM as long as overlap state is
carried forward.

## Loss

The initial non-adversarial objective combines:

- log-magnitude L1;
- magnitude-weighted circular phase distance;
- reconstructed waveform L1.

A later quality phase can add multi-resolution STFT and discriminator feature
matching without changing the deployed generator architecture.

## Teacher/distillation strategy

Vocos remains the quality oracle. Training can use both:

1. ground-truth waveform spectral targets;
2. optional Vocos reconstruction outputs as a teacher for perceptual matching.

The final mobile runtime does not require Vocos.
