from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TimbreNetConfig:
    """Compact target-speaker encoder for WavLM+ECAPA distillation."""

    input_dim: int = 80
    hidden_dim: int = 160
    embedding_dim: int = 256
    kernel_size: int = 5
    attention_dim: int = 128
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 1, 2, 4)

    def __post_init__(self) -> None:
        if min(
            self.input_dim,
            self.hidden_dim,
            self.embedding_dim,
            self.attention_dim,
        ) <= 0:
            raise ValueError("dimensions must be positive")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least 3")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must contain positive integers")


class TimbreBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
            bias=False,
        )
        self.mix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.layer_scale = nn.Parameter(torch.full((hidden_dim,), 0.1))

    def forward(self, hidden: Tensor) -> Tensor:
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.depthwise(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.mix(hidden)
        return residual + hidden * self.layer_scale


class AttentiveStatisticsPool(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, hidden: Tensor, mask: Tensor | None = None) -> Tensor:
        scores = self.attention(hidden).squeeze(-1)
        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError(
                    f"expected mask shape {tuple(scores.shape)}, got {tuple(mask.shape)}"
                )
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = torch.sum(weights * hidden, dim=1)
        centered = hidden - mean.unsqueeze(1)
        variance = torch.sum(weights * centered.square(), dim=1)
        std = torch.sqrt(torch.clamp(variance, min=1e-5))
        return torch.cat((mean, std), dim=-1)


class TimbreNet(nn.Module):
    """~1M parameter utterance-level target-speaker encoder.

    Input:
        [batch, frames, 80] log-mel features.

    Output:
        [batch, 256] L2-normalized global speaker embeddings.

    Reference-audio encoding is not latency-critical, so the temporal blocks
    use symmetric context rather than causal padding. This provides better
    timbre aggregation while keeping the live source path fully causal.
    """

    def __init__(self, config: TimbreNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or TimbreNetConfig()

        self.input_projection = nn.Linear(
            self.config.input_dim,
            self.config.hidden_dim,
        )
        self.blocks = nn.ModuleList(
            TimbreBlock(
                self.config.hidden_dim,
                self.config.kernel_size,
                dilation,
            )
            for dilation in self.config.dilations
        )
        self.output_norm = nn.LayerNorm(self.config.hidden_dim)
        self.pool = AttentiveStatisticsPool(
            self.config.hidden_dim,
            self.config.attention_dim,
        )
        self.output_projection = nn.Linear(
            self.config.hidden_dim * 2,
            self.config.embedding_dim,
        )

    def forward(
        self,
        features: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, frames, input_dim]")
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected input_dim={self.config.input_dim}, got {features.shape[-1]}"
            )
        if features.shape[1] == 0:
            raise ValueError("reference must contain at least one frame")

        hidden = self.input_projection(features)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.output_norm(hidden)
        pooled = self.pool(hidden, mask)
        embedding = self.output_projection(pooled)
        return F.normalize(embedding, dim=-1, eps=1e-8)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
