from __future__ import annotations

import torch
from condition_fusion import ConditionFusion, count_parameters


def main() -> int:
    torch.manual_seed(23)
    model = ConditionFusion().eval()

    content = torch.randn(2, 37, 256)
    pitch = torch.randn(2, 37, 3)
    speaker = torch.randn(2, 256)
    speaker = torch.nn.functional.normalize(speaker, dim=-1)

    with torch.no_grad():
        fused, attention = model(content, pitch, speaker)

    parameters = count_parameters(model)
    attention_error = float((attention.sum(dim=-1) - 1.0).abs().max())

    print(f"parameters={parameters}")
    print(f"fp32_mebibytes={parameters * 4 / 1024 / 1024:.3f}")
    print(f"fused_shape={tuple(fused.shape)}")
    print(f"attention_shape={tuple(attention.shape)}")
    print(f"max_attention_sum_error={attention_error:.9f}")

    if fused.shape != (2, 37, 256):
        raise RuntimeError("unexpected fused output shape")
    if attention.shape != (2, 37, 8):
        raise RuntimeError("unexpected timbre-attention shape")
    if attention_error > 1e-6:
        raise RuntimeError("timbre attention does not sum to one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
