from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ContentDistillationLossConfig:
    stride: int = 4
    offset: int = 3
    mse_weight: float = 1.0
    cosine_weight: float = 0.5
    temporal_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not 0 <= self.offset < self.stride:
            raise ValueError("offset must satisfy 0 <= offset < stride")
        if min(self.mse_weight, self.cosine_weight, self.temporal_weight) < 0.0:
            raise ValueError("loss weights must be non-negative")


def content_distillation_loss(
    dense_student: Tensor,
    teacher_bn: Tensor,
    config: ContentDistillationLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Distill dense student features against 40 ms teacher BN features.

    The student predicts at the acoustic frontend's 10 ms cadence. Only the
    cadence-aligned frames are supervised against the teacher, keeping the
    exported model simple while the runtime performs cheap frame selection.
    """

    cfg = config or ContentDistillationLossConfig()
    student = dense_student[:, cfg.offset :: cfg.stride]

    if student.shape != teacher_bn.shape:
        raise ValueError(
            "student/teacher shape mismatch after cadence selection: "
            f"{tuple(student.shape)} != {tuple(teacher_bn.shape)}"
        )

    mse = F.mse_loss(student, teacher_bn)
    cosine = 1.0 - F.cosine_similarity(student, teacher_bn, dim=-1).mean()

    if student.shape[1] > 1:
        student_delta = student[:, 1:] - student[:, :-1]
        teacher_delta = teacher_bn[:, 1:] - teacher_bn[:, :-1]
        temporal = F.smooth_l1_loss(student_delta, teacher_delta)
    else:
        temporal = torch.zeros((), device=student.device, dtype=student.dtype)

    total = (
        cfg.mse_weight * mse
        + cfg.cosine_weight * cosine
        + cfg.temporal_weight * temporal
    )
    return total, {
        "mse": mse.detach(),
        "cosine": cosine.detach(),
        "temporal": temporal.detach(),
        "total": total.detach(),
    }
