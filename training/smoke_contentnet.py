from __future__ import annotations

import torch
from contentnet import CausalContentNet, count_parameters


def main() -> int:
    torch.manual_seed(7)
    model = CausalContentNet().eval()
    features = torch.randn(2, 101, 80)

    with torch.no_grad():
        offline = model(features)
        cache = model.initial_cache(features.shape[0])
        outputs = []
        cursor = 0

        for chunk_size in (1, 7, 3, 19, 2, 31, 5, 33):
            if cursor >= features.shape[1]:
                break
            end = min(features.shape[1], cursor + chunk_size)
            output, cache = model.forward_chunk(features[:, cursor:end], cache)
            outputs.append(output)
            cursor = end

        if cursor < features.shape[1]:
            output, cache = model.forward_chunk(features[:, cursor:], cache)
            outputs.append(output)

        streaming = torch.cat(outputs, dim=1)

    max_error = float((offline - streaming).abs().max())
    parameters = count_parameters(model)
    print(f"parameters={parameters}")
    print(f"fp32_mebibytes={parameters * 4 / 1024 / 1024:.3f}")
    print(f"max_streaming_error={max_error:.9f}")

    if max_error > 2e-6:
        raise RuntimeError("streaming output diverged from offline output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
