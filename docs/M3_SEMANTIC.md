# M3 semantic sidecar

M3 gives Voxera a semantic/discourse path without putting speech recognition on
the audio critical path.

## Core rule: ASR never runs inside conversion

The converter consumes only an immutable `SemanticSnapshot` that was already
produced by a background sidecar.

```text
microphone/source PCM
       │
       ├──────────────────────────> real-time VC graph
       │                                │
       │                                └─ reads latest semantic snapshot
       │
       └─ background worker
              ↓
           ASR backend
              ↓
       TranscriptHypothesis
              ↓
      NativeSemanticAnalyzer
              ↓
        SemanticMailbox
```

If ASR stalls, produces no update, or is disabled, the voice-conversion path
continues. It never waits on transcription.

## Semantic snapshot

The first runtime descriptor is 16-dimensional:

```text
0  statement
1  question
2  command
3  emphasis
4  hesitation
5  uncertainty
6  negation
7  continuation / unfinished phrase
8  punctuation density
9  repetition
10 lexical emphasis
11 normalized text length
12 Han-script ratio
13 Latin-script ratio
14 acoustic rising-contour cue
15 acoustic emphasis cue
```

The first eight channels participate in a coarse discourse-mode decision. The
analyzer deliberately does **not** claim that text alone can reliably determine
sarcasm, happiness, anger, or other hidden affect.

When an M2 `ProsodyTrack` is available, rising contour and acoustic emphasis
are fused into the semantic snapshot. This is how Voxera begins combining
"what the sentence appears to be doing" with "how it was actually spoken".

## Streaming ASR contract

`StreamingASRBackend` is intentionally tiny:

- `reset()`
- `accept_audio(samples, sample_rate)`
- `finish()`

Backends return versioned `TranscriptHypothesis` objects. The core does not
depend on a specific ASR implementation.

### Primary target: sherpa-onnx

The first production adapter target is sherpa-onnx because it has:

- offline and streaming ASR;
- Android and desktop deployment;
- bilingual Chinese/English streaming models;
- Qualcomm QNN/HTP deployment paths;
- Apache-2.0 source licensing.

Voxera pins source bootstrap to `v1.13.8`. Model licenses are separate and
must be checked per selected model.

The Python reference adapter is implemented as
`SherpaOnnxStreamingASR` for streaming transducer models. It wraps the official
`OnlineRecognizer.from_transducer` API and maps partial/final endpoint results
to Voxera's versioned `TranscriptHypothesis` contract.

Desktop development can install the optional CPU package group with:

```bash
pip install -e ".[asr-sherpa]"
```

Android production should use sherpa-onnx's native/JNI/QNN artifacts instead of
shipping Python.

### Compatibility target: whisper.cpp

whisper.cpp is implemented as a **phrase-final quality/fallback adapter** rather
than pretending to have the same latency contract as sherpa-onnx.

`WhisperCppCliASR` buffers 16 kHz PCM on the background sidecar and invokes the
official `whisper-cli` only on `finish()`. It reads the CLI's `-otxt`
output file and publishes one final `TranscriptHypothesis`.

This gives Voxera a useful second engine for phrase-final quality checks while
keeping the real-time path honest: sherpa-onnx owns partial hypotheses; whisper
does not block audio and does not manufacture fake partials.

whisper.cpp provides:

- C/C++ core;
- Android, Windows, macOS, Linux support;
- integer quantization;
- streaming examples for future native work;
- MIT source license.

Voxera pins source bootstrap to `v1.9.4`.

Bootstrap source trees without downloading model weights:

```bash
sh tools/bootstrap_asr.sh
```

Default: sherpa-onnx only.

Optional environment configuration:

```text
VOXERA_ASR_BACKENDS=sherpa
VOXERA_ASR_BACKENDS=whisper
VOXERA_ASR_BACKENDS=both
```

## Latest-value mailbox

`SemanticMailbox` keeps only the newest immutable snapshot.

The intended production ownership is:

```text
ASR/background thread: publish(snapshot)
audio thread:          read_latest()
```

The Python reference uses immutable object replacement. A future C++ runtime
will use an atomic latest-snapshot pointer/ring-slot rather than a blocking
queue.

## Semantic conditioning

M3 adds `SemanticConditioner` after M1/M2 condition construction:

```text
M1 fused condition
      ↓
optional M2 ProsodyConditioner
      ↓
optional M3 SemanticConditioner
      ↓
DecoderNet
```

The semantic adapter receives the 16-d snapshot plus its confidence. Its output
projection is zero-initialized, so a fresh M3 model is an exact identity
transform.

This preserves the same rollout invariant used by M2: adding architecture before
training must not silently alter M1/M2 output.

## What is implemented now

- backend-neutral streaming ASR protocol;
- transcript revision/finality contract;
- latest-value semantic mailbox;
- Chinese/English/mixed-script heuristic language cue;
- deterministic intent/discourse descriptor;
- optional M2 acoustic-prosody fusion;
- zero-safe neural semantic conditioner;
- optional NativeOfflinePipeline semantic snapshot input;
- tests proving that semantic conditioning is not required for duration/output
  contracts;
- concrete sherpa-onnx Python streaming adapter with endpoint/reset/finalization
  lifecycle coverage.

## What still waits on real models/data

- native sherpa-onnx JNI/C++ adapter matching the implemented Python reference;
- concrete whisper.cpp adapter;
- ASR model selection/benchmarking;
- learned intent/style encoder beyond deterministic pseudo-targets;
- semantic-to-prosody target supervision;
- partial-hypothesis stabilization tuned on real speech;
- latency/power measurements on Android hardware.


## Explicit backend selection

M3 provides `ASRBackendSelection` and `ASRBackendKind`.

Selection is explicit:

```text
SHERPA  → low-latency streaming partial/final hypotheses
WHISPER → buffered phrase-final transcription
```

Voxera deliberately does not implement a hidden "try sherpa, silently fall back
to whisper" policy. The two engines have materially different latency and power
characteristics, so fallback must be an application-level decision.

The factory `create_asr_backend()` validates that a backend kind and its config
type match before constructing the engine.
