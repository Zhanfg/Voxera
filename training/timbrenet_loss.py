from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TimbreDistillationLossConfig:
    cosine_weight: float = 1.0
    mse_weight: float = 0.25
    consistency_weight: float = 0.1

    def __post_init__(self) -> None:
        if min(
            self.cosine_weight,
            self.mse_weight,
            self.consistency_weight,
        ) < 0.0:
            raise ValueError("loss weights must be non-negative")


def timbre_distillation_loss(
    student: Tensor,
    teacher: Tensor,
    *,
    second_view: Tensor | None = None,
    config: TimbreDistillationLossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Distill MeanVC2 speaker embeddings into the compact TimbreNet space."""

    cfg = config or TimbreDistillationLossConfig()
    if student.shape != teacher.shape:
        raise ValueError(
            f"student/teacher shape mismatch: {tuple(student.shape)} != {tuple(teacher.shape)}"
        )

    student_n = F.normalize(student, dim=-1, eps=1e-8)
    teacher_n = F.normalize(teacher, dim=-1, eps=1e-8)

    cosine = 1.0 - F.cosine_similarity(student_n, teacher_n, dim=-1).mean()
    mse = F.mse_loss(student_n, teacher_n)

    if second_view is None:
        consistency = torch.zeros((), device=student.device, dtype=student.dtype)
    else:
        if second_view.shape != student.shape:
            raise ValueError("second_view must have the same shape as student")
        second_n = F.normalize(second_view, dim=-1, eps=1e-8)
        consistency = 1.0 - F.cosine_similarity(
            student_n,
            second_n,
            dim=-1,
        ).mean()

    total = (
        cfg.cosine_weight * cosine
        + cfg.mse_weight * mse
        + cfg.consistency_weight * consistency
    )
    return total, {
        "cosine": cosine.detach(),
        "mse": mse.detach(),
        "consistency": consistency.detach(),
        "total": total.detach(),
    }
