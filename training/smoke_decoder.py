from __future__ import annotations

import torch
from decoder import DecoderNet, count_parameters


def main() -> int:
    torch.manual_seed(29)
    model = DecoderNet().eval()
    conditions = torch.randn(2, 41, 256)

    with torch.no_grad():
        offline = model(conditions)
        cache = model.initial_cache(conditions.shape[0])
        outputs = []
        cursor = 0

        for chunk_size in (1, 5, 2, 11, 3, 7):
            if cursor >= conditions.shape[1]:
                break
            end = min(conditions.shape[1], cursor + chunk_size)
            output, cache = model.forward_chunk(
                conditions[:, cursor:end],
                cache,
            )
            outputs.append(output)
            cursor = end

        if cursor < conditions.shape[1]:
            output, cache = model.forward_chunk(conditions[:, cursor:], cache)
            outputs.append(output)

        streaming = torch.cat(outputs, dim=1)

    parameters = count_parameters(model)
    max_error = float((offline - streaming).abs().max())

    print(f"parameters={parameters}")
    print(f"fp32_mebibytes={parameters * 4 / 1024 / 1024:.3f}")
    print(f"output_shape={tuple(offline.shape)}")
    print(f"max_streaming_error={max_error:.9f}")

    if offline.shape != (2, 164, 80):
        raise RuntimeError("unexpected DecoderNet output shape")
    if max_error > 2e-6:
        raise RuntimeError("streaming DecoderNet diverged from offline output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
