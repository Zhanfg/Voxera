from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ProsodyLossConfig:
    local_descriptor_weight: float = 1.0
    global_descriptor_weight: float = 1.0
    local_smoothness_weight: float = 0.05
    global_norm_weight: float = 0.01

    def __post_init__(self) -> None:
        if min(
            self.local_descriptor_weight,
            self.global_descriptor_weight,
            self.local_smoothness_weight,
            self.global_norm_weight,
        ) < 0.0:
            raise ValueError("loss weights must be non-negative")


def prosody_descriptor_loss(
    local_embedding: Tensor,
    global_embedding: Tensor,
    local_prediction: Tensor,
    global_prediction: Tensor,
    local_target: Tensor,
    global_target: Tensor,
    config: ProsodyLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Bootstrap ProsodyNet from deterministic acoustic pseudo-targets."""

    cfg = config or ProsodyLossConfig()
    if local_prediction.shape != local_target.shape:
        raise ValueError("local descriptor prediction/target shapes must match")
    if global_prediction.shape != global_target.shape:
        raise ValueError("global descriptor prediction/target shapes must match")

    local_loss = F.smooth_l1_loss(local_prediction, local_target)
    global_loss = F.smooth_l1_loss(global_prediction, global_target)

    if local_embedding.shape[1] > 1:
        delta = local_embedding[:, 1:] - local_embedding[:, :-1]
        smoothness = delta.square().mean()
    else:
        smoothness = torch.zeros(
            (),
            device=local_embedding.device,
            dtype=local_embedding.dtype,
        )

    global_norm = (global_embedding.square().mean(dim=-1) - 1.0).abs().mean()

    total = (
        cfg.local_descriptor_weight * local_loss
        + cfg.global_descriptor_weight * global_loss
        + cfg.local_smoothness_weight * smoothness
        + cfg.global_norm_weight * global_norm
    )
    return total, {
        "local_descriptor": local_loss.detach(),
        "global_descriptor": global_loss.detach(),
        "local_smoothness": smoothness.detach(),
        "global_norm": global_norm.detach(),
        "total": total.detach(),
    }
