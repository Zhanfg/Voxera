from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class DecoderLossConfig:
    mel_weight: float = 1.0
    delta_weight: float = 0.5
    acceleration_weight: float = 0.25

    def __post_init__(self) -> None:
        if min(
            self.mel_weight,
            self.delta_weight,
            self.acceleration_weight,
        ) < 0.0:
            raise ValueError("loss weights must be non-negative")


def decoder_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor | None = None,
    config: DecoderLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Reconstruct target log-mel frames while preserving local dynamics."""

    cfg = config or DecoderLossConfig()
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "prediction and target must have matching [batch, frames, mels] shapes"
        )

    mel = _masked_l1(prediction, target, mask)

    if prediction.shape[1] > 1:
        pred_delta = prediction[:, 1:] - prediction[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        delta_mask = None if mask is None else mask[:, 1:] & mask[:, :-1]
        delta = _masked_l1(pred_delta, target_delta, delta_mask)
    else:
        delta = torch.zeros((), device=prediction.device, dtype=prediction.dtype)

    if prediction.shape[1] > 2:
        pred_accel = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
        target_accel = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
        accel_mask = None
        if mask is not None:
            accel_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
        acceleration = _masked_l1(pred_accel, target_accel, accel_mask)
    else:
        acceleration = torch.zeros(
            (),
            device=prediction.device,
            dtype=prediction.dtype,
        )

    total = (
        cfg.mel_weight * mel
        + cfg.delta_weight * delta
        + cfg.acceleration_weight * acceleration
    )
    return total, {
        "mel": mel.detach(),
        "delta": delta.detach(),
        "acceleration": acceleration.detach(),
        "total": total.detach(),
    }


def _masked_l1(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None,
) -> Tensor:
    if mask is None:
        return F.l1_loss(prediction, target)

    if mask.ndim != 2 or mask.shape != prediction.shape[:2]:
        raise ValueError("mask must have shape [batch, frames]")
    weights = mask.to(dtype=prediction.dtype).unsqueeze(-1)
    denominator = weights.sum() * prediction.shape[-1]
    if float(denominator.detach()) <= 0.0:
        return torch.zeros((), device=prediction.device, dtype=prediction.dtype)

    error = torch.abs(prediction - target) * weights
    return error.sum() / denominator
