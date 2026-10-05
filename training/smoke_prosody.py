from __future__ import annotations

import torch

from training.prosody_conditioner import ProsodyConditioner, count_parameters as conditioner_parameters
from training.prosodynet import ProsodyNet, count_parameters as prosody_parameters


def main() -> int:
    torch.manual_seed(79)
    model = ProsodyNet().eval()

    acoustic = torch.randn(1, 53, 80)
    pitch = torch.randn(1, 53, 3)

    with torch.no_grad():
        full_local, global_style, full_descriptor, global_descriptor = model(
            acoustic,
            pitch,
        )

        cache = model.initial_cache(1)
        local_parts = []
        descriptor_parts = []
        start = 0
        for size in (7, 11, 5, 13, 17):
            end = min(start + size, acoustic.shape[1])
            if end <= start:
                break
            local, descriptor, cache = model.forward_chunk(
                acoustic[:, start:end],
                pitch[:, start:end],
                cache,
            )
            local_parts.append(local)
            descriptor_parts.append(descriptor)
            start = end

        if start < acoustic.shape[1]:
            local, descriptor, cache = model.forward_chunk(
                acoustic[:, start:],
                pitch[:, start:],
                cache,
            )
            local_parts.append(local)
            descriptor_parts.append(descriptor)

        streamed_local = torch.cat(local_parts, dim=1)
        streamed_descriptor = torch.cat(descriptor_parts, dim=1)

    local_error = float((full_local - streamed_local).abs().max())
    descriptor_error = float((full_descriptor - streamed_descriptor).abs().max())
    if local_error > 1e-5:
        raise RuntimeError(f"ProsodyNet local streaming mismatch: {local_error}")
    if descriptor_error > 1e-5:
        raise RuntimeError(
            f"ProsodyNet descriptor streaming mismatch: {descriptor_error}"
        )

    if tuple(global_style.shape) != (1, 16):
        raise RuntimeError("unexpected global prosody embedding shape")
    if tuple(global_descriptor.shape) != (1, 8):
        raise RuntimeError("unexpected global prosody descriptor shape")

    conditioner = ProsodyConditioner().eval()
    base = torch.randn(1, 13, 256)
    local_40 = full_local[:, 3::4][:, :13]
    if local_40.shape[1] != base.shape[1]:
        base = base[:, : local_40.shape[1]]

    with torch.no_grad():
        conditioned = conditioner(base, local_40, global_style)

    identity_error = float((conditioned - base).abs().max())
    if identity_error != 0.0:
        raise RuntimeError(
            "fresh ProsodyConditioner must preserve the M1 condition exactly"
        )

    print(f"prosodynet_parameters={prosody_parameters(model)}")
    print(f"conditioner_parameters={conditioner_parameters(conditioner)}")
    print(f"local_streaming_max_error={local_error:.9f}")
    print(f"descriptor_streaming_max_error={descriptor_error:.9f}")
    print(f"zero_safe_identity_error={identity_error:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
