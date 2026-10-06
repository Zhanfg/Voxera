from __future__ import annotations

import torch
from semantic_conditioner import SemanticConditioner, count_parameters


def main() -> int:
    torch.manual_seed(97)
    model = SemanticConditioner().eval()

    base = torch.randn(2, 19, 256)
    semantic = torch.randn(2, 16)
    confidence = torch.tensor([0.95, 0.40])

    with torch.no_grad():
        output = model(base, semantic, confidence)

    identity_error = float((output - base).abs().max())
    if identity_error != 0.0:
        raise RuntimeError(
            "fresh SemanticConditioner must preserve M1/M2 conditions exactly"
        )

    print(f"parameters={count_parameters(model)}")
    print(f"identity_error={identity_error:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
