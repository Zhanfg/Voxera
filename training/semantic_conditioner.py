from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class SemanticConditionerConfig:
    base_dim: int = 256
    semantic_dim: int = 16
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        if min(self.base_dim, self.semantic_dim, self.hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")


class SemanticConditioner(nn.Module):
    """Zero-safe residual adapter from semantic snapshot to decoder condition.

    The final projection is initialized to zero. Before training, adding M3 to
    the graph is therefore an exact identity transform.
    """

    def __init__(self, config: SemanticConditionerConfig | None = None) -> None:
        super().__init__()
        self.config = config or SemanticConditionerConfig()
        cfg = self.config

        self.encoder = nn.Sequential(
            nn.Linear(cfg.semantic_dim + 1, cfg.hidden_dim),
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
        semantic_features: Tensor,
        semantic_confidence: Tensor,
    ) -> Tensor:
        cfg = self.config
        if base.ndim != 3 or base.shape[-1] != cfg.base_dim:
            raise ValueError(
                f"base must have shape [batch, frames, {cfg.base_dim}]"
            )
        if semantic_features.ndim != 2 or semantic_features.shape != (
            base.shape[0],
            cfg.semantic_dim,
        ):
            raise ValueError(
                "semantic_features must have shape "
                f"[batch, {cfg.semantic_dim}]"
            )
        if semantic_confidence.ndim == 1:
            semantic_confidence = semantic_confidence.unsqueeze(-1)
        if semantic_confidence.shape != (base.shape[0], 1):
            raise ValueError("semantic_confidence must have shape [batch, 1]")

        confidence = torch.clamp(semantic_confidence, 0.0, 1.0)
        hidden = self.encoder(
            torch.cat((semantic_features, confidence), dim=-1)
        )
        delta = self.delta_projection(hidden).unsqueeze(1)
        gate = torch.sigmoid(self.gate_projection(hidden)).unsqueeze(1)
        return base + confidence.unsqueeze(1) * gate * delta


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
