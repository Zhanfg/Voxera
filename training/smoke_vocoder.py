from __future__ import annotations

import torch
from lite_vocoder import LiteVocoder, count_parameters


def main() -> int:
    torch.manual_seed(31)
    model = LiteVocoder().eval()
    mel = torch.randn(2, 97, 80)

    with torch.no_grad():
        offline_logmag, offline_phase = model(mel)
        cache = model.initial_cache(mel.shape[0])
        logmag_parts = []
        phase_parts = []
        cursor = 0

        for chunk_size in (1, 8, 3, 21, 5, 34):
            if cursor >= mel.shape[1]:
                break
            end = min(mel.shape[1], cursor + chunk_size)
            logmag, phase, cache = model.forward_chunk(
                mel[:, cursor:end],
                cache,
            )
            logmag_parts.append(logmag)
            phase_parts.append(phase)
            cursor = end

        if cursor < mel.shape[1]:
            logmag, phase, cache = model.forward_chunk(mel[:, cursor:], cache)
            logmag_parts.append(logmag)
            phase_parts.append(phase)

        streaming_logmag = torch.cat(logmag_parts, dim=1)
        streaming_phase = torch.cat(phase_parts, dim=1)

    parameters = count_parameters(model)
    magnitude_error = float((offline_logmag - streaming_logmag).abs().max())
    phase_error = float((offline_phase - streaming_phase).abs().max())

    print(f"parameters={parameters}")
    print(f"fp32_mebibytes={parameters * 4 / 1024 / 1024:.3f}")
    print(f"spectral_shape={tuple(offline_logmag.shape)}")
    print(f"max_logmag_streaming_error={magnitude_error:.9f}")
    print(f"max_phase_streaming_error={phase_error:.9f}")

    if offline_logmag.shape != (2, 97, 161):
        raise RuntimeError("unexpected LiteVocoder spectral shape")
    if max(magnitude_error, phase_error) > 2e-6:
        raise RuntimeError("streaming LiteVocoder diverged from offline output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
