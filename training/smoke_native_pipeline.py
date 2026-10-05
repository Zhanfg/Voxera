from __future__ import annotations

import numpy as np
import torch
from condition_fusion import ConditionFusion
from contentnet import CausalContentNet
from decoder import DecoderNet
from lite_vocoder import LiteVocoder
from timbrenet import TimbreNet

from voxera.speaker import SpeakerEmbedding
from voxera.vocoder import SpectralFrames, StreamingISTFT


def main() -> int:
    torch.manual_seed(37)

    content_net = CausalContentNet().eval()
    timbre_net = TimbreNet().eval()
    fusion = ConditionFusion().eval()
    decoder = DecoderNet().eval()
    vocoder = LiteVocoder().eval()

    source_fbank = torch.randn(1, 98, 80)
    target_fbank = torch.randn(1, 180, 80)

    with torch.no_grad():
        dense_content = content_net(source_fbank)
        speaker = timbre_net(target_fbank)

        common_frames = 97
        indices = torch.arange(3, common_frames, 4)
        content_40ms = dense_content[:, indices]

        pitch = torch.zeros(1, indices.numel(), 3)
        pitch[:, :, 0] = 1.0
        pitch[:, :, 1] = 1.0
        pitch[:, :, 2] = 0.9

        fused, attention = fusion(content_40ms, pitch, speaker)
        mel = decoder(fused)
        log_magnitude, phase = vocoder(mel)

    spectral = SpectralFrames(
        log_magnitude[0].cpu().numpy().astype(np.float32),
        phase[0].cpu().numpy().astype(np.float32),
    )
    pcm = StreamingISTFT().push(spectral)

    expected_condition_frames = indices.numel()
    expected_mel_frames = expected_condition_frames * 4
    expected_samples = expected_mel_frames * 160

    print(f"content_shape={tuple(dense_content.shape)}")
    print(f"speaker_shape={tuple(speaker.shape)}")
    print(f"condition_shape={tuple(fused.shape)}")
    print(f"mel_shape={tuple(mel.shape)}")
    print(f"spectral_shape={tuple(log_magnitude.shape)}")
    print(f"pcm_samples={pcm.size}")
    print(f"attention_max_error={float((attention.sum(-1) - 1.0).abs().max()):.9f}")
    print(f"speaker_norm={float(speaker.norm(dim=-1).item()):.9f}")
    print(f"pcm_finite={bool(np.all(np.isfinite(pcm)))}")

    if speaker.shape != (1, 256):
        raise RuntimeError("unexpected speaker shape")
    if fused.shape != (1, expected_condition_frames, 256):
        raise RuntimeError("unexpected fused condition shape")
    if mel.shape != (1, expected_mel_frames, 80):
        raise RuntimeError("unexpected decoder mel shape")
    if log_magnitude.shape != (1, expected_mel_frames, 161):
        raise RuntimeError("unexpected vocoder spectral shape")
    if pcm.shape != (expected_samples,):
        raise RuntimeError("unexpected PCM length")
    if not np.all(np.isfinite(pcm)):
        raise RuntimeError("native pipeline produced non-finite PCM")
    if abs(float(speaker.norm(dim=-1).item()) - 1.0) > 1e-5:
        raise RuntimeError("speaker embedding is not normalized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
