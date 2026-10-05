from __future__ import annotations

import torch
from timbrenet import TimbreNet, count_parameters


def main() -> int:
    torch.manual_seed(19)
    model = TimbreNet().eval()
    features = torch.randn(3, 250, 80)

    with torch.no_grad():
        embeddings = model(features)

    parameters = count_parameters(model)
    norms = embeddings.norm(dim=-1)
    print(f"parameters={parameters}")
    print(f"fp32_mebibytes={parameters * 4 / 1024 / 1024:.3f}")
    print(f"output_shape={tuple(embeddings.shape)}")
    print(f"max_norm_error={float((norms - 1.0).abs().max()):.9f}")

    if embeddings.shape != (3, 256):
        raise RuntimeError("unexpected TimbreNet output shape")
    if float((norms - 1.0).abs().max()) > 1e-5:
        raise RuntimeError("TimbreNet embeddings are not L2-normalized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
