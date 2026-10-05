from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ProsodyConditionerConfig:
    base_dim: int = 256
    local_dim: int = 32
    global_dim: int = 16
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        if min(
            self.base_dim,
            self.local_dim,
            self.global_dim,
            self.hidden_dim,
        ) <= 0:
            raise ValueError("all dimensions must be positive")


class ProsodyConditioner(nn.Module):
    """Residual prosody adapter placed after the stable M1 fusion layer.

    The final projection is zero-initialized. A freshly constructed M2 adapter
    therefore reproduces the M1 condition exactly until prosody training learns
    a useful residual.
    """

    def __init__(self, config: ProsodyConditionerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProsodyConditionerConfig()
        cfg = self.config

        self.input_projection = nn.Sequential(
            nn.Linear(cfg.local_dim + cfg.global_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.delta_projection = nn.Linear(cfg.hidden_dim, cfg.base_dim)
        self.gate_projection = nn.Linear(cfg.hidden_dim, cfg.base_dim)

        nn.init.zeros_(self.delta_projection.weight)
        nn.init.zeros_(self.delta_projection.bias)
        nn.init.zeros_(self.gate_projection.weight)
        nn.init.constant_(self.gate_projection.bias, -2.0)

    def forward(
        self,
        base: Tensor,
        local_prosody: Tensor,
        global_style: Tensor,
    ) -> Tensor:
        cfg = self.config
        if base.ndim != 3 or base.shape[-1] != cfg.base_dim:
            raise ValueError(
                f"base must have shape [batch, frames, {cfg.base_dim}]"
            )
        if local_prosody.ndim != 3 or local_prosody.shape != (
            base.shape[0],
            base.shape[1],
            cfg.local_dim,
        ):
            raise ValueError(
                "local_prosody must share base batch/frame dimensions and "
                f"have {cfg.local_dim} channels"
            )
        if global_style.ndim != 2 or global_style.shape != (
            base.shape[0],
            cfg.global_dim,
        ):
            raise ValueError(
                f"global_style must have shape [batch, {cfg.global_dim}]"
            )

        repeated_global = global_style.unsqueeze(1).expand(
            -1,
            base.shape[1],
            -1,
        )
        hidden = self.input_projection(
            torch.cat((local_prosody, repeated_global), dim=-1)
        )
        delta = self.delta_projection(hidden)
        gate = torch.sigmoid(self.gate_projection(hidden))
        return base + gate * delta


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
