from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ConditionFusionConfig:
    content_dim: int = 256
    speaker_dim: int = 256
    pitch_dim: int = 3
    token_count: int = 8
    token_dim: int = 128
    pitch_hidden_dim: int = 64
    output_dim: int = 256

    def __post_init__(self) -> None:
        if min(
            self.content_dim,
            self.speaker_dim,
            self.pitch_dim,
            self.token_count,
            self.token_dim,
            self.pitch_hidden_dim,
            self.output_dim,
        ) <= 0:
            raise ValueError("all dimensions must be positive")


class TimbreTokenProjector(nn.Module):
    """Expand one global speaker vector into a small learned token bank."""

    def __init__(self, config: ConditionFusionConfig) -> None:
        super().__init__()
        self.config = config
        self.seed = nn.Linear(
            config.speaker_dim,
            config.token_count * config.token_dim,
        )
        self.slot_bias = nn.Parameter(
            torch.zeros(config.token_count, config.token_dim)
        )
        self.key_projection = nn.Linear(config.token_dim, config.token_dim)
        self.value_projection = nn.Linear(config.token_dim, config.token_dim)

    def forward(self, speaker: Tensor) -> tuple[Tensor, Tensor]:
        if speaker.ndim != 2 or speaker.shape[-1] != self.config.speaker_dim:
            raise ValueError(
                "speaker must have shape "
                f"[batch, {self.config.speaker_dim}]"
            )

        batch = speaker.shape[0]
        tokens = self.seed(speaker).reshape(
            batch,
            self.config.token_count,
            self.config.token_dim,
        )
        tokens = tokens + self.slot_bias.unsqueeze(0)
        return self.key_projection(tokens), self.value_projection(tokens)


class ConditionFusion(nn.Module):
    """Fuse linguistic content, target timbre, and source pitch conditions.

    The timbre path is content-conditioned: ContentNet features query a small
    token bank derived from the global speaker embedding. This preserves the
    separation between identity extraction (TimbreNet) and fine-grained timbre
    retrieval.
    """

    def __init__(self, config: ConditionFusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConditionFusionConfig()
        cfg = self.config

        self.timbre_tokens = TimbreTokenProjector(cfg)
        self.query_projection = nn.Linear(cfg.content_dim, cfg.token_dim)
        self.pitch_projection = nn.Sequential(
            nn.Linear(cfg.pitch_dim, cfg.pitch_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.pitch_hidden_dim, cfg.pitch_hidden_dim),
        )

        fused_dim = cfg.content_dim + cfg.token_dim + cfg.pitch_hidden_dim
        self.fused_candidate = nn.Sequential(
            nn.Linear(fused_dim, cfg.output_dim),
            nn.SiLU(),
            nn.Linear(cfg.output_dim, cfg.output_dim),
        )
        self.fused_gate = nn.Linear(fused_dim, cfg.output_dim)
        self.content_residual = (
            nn.Identity()
            if cfg.content_dim == cfg.output_dim
            else nn.Linear(cfg.content_dim, cfg.output_dim)
        )
        self.output_norm = nn.LayerNorm(cfg.output_dim)

    def forward(
        self,
        content: Tensor,
        pitch: Tensor,
        speaker: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        if content.ndim != 3 or content.shape[-1] != cfg.content_dim:
            raise ValueError(
                f"content must have shape [batch, frames, {cfg.content_dim}]"
            )
        if pitch.ndim != 3 or pitch.shape[-1] != cfg.pitch_dim:
            raise ValueError(
                f"pitch must have shape [batch, frames, {cfg.pitch_dim}]"
            )
        if content.shape[:2] != pitch.shape[:2]:
            raise ValueError("content and pitch must share batch/frame dimensions")
        if speaker.ndim != 2 or speaker.shape != (
            content.shape[0],
            cfg.speaker_dim,
        ):
            raise ValueError(
                f"speaker must have shape [batch, {cfg.speaker_dim}]"
            )

        keys, values = self.timbre_tokens(speaker)
        queries = self.query_projection(content)

        scale = 1.0 / sqrt(cfg.token_dim)
        scores = torch.matmul(queries, keys.transpose(1, 2)) * scale
        attention = torch.softmax(scores, dim=-1)
        timbre = torch.matmul(attention, values)

        pitch_hidden = self.pitch_projection(pitch)
        fused_input = torch.cat((content, timbre, pitch_hidden), dim=-1)
        candidate = self.fused_candidate(fused_input)
        gate = torch.sigmoid(self.fused_gate(fused_input))
        residual = self.content_residual(content)
        fused = self.output_norm(residual + gate * candidate)
        return fused, attention


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
