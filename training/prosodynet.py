from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ProsodyNetConfig:
    acoustic_dim: int = 80
    pitch_dim: int = 3
    hidden_dim: int = 128
    local_dim: int = 32
    global_dim: int = 16
    descriptor_dim: int = 8
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2, 4, 8)

    def __post_init__(self) -> None:
        if min(
            self.acoustic_dim,
            self.pitch_dim,
            self.hidden_dim,
            self.local_dim,
            self.global_dim,
            self.descriptor_dim,
        ) <= 0:
            raise ValueError("all dimensions must be positive")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must contain positive integers")

    @property
    def max_cache_frames(self) -> int:
        return max(self.dilations) * (self.kernel_size - 1)


class CausalProsodyBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        max_cache_frames: int,
    ) -> None:
        super().__init__()
        self.context_frames = dilation * (kernel_size - 1)
        self.max_cache_frames = max_cache_frames
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size,
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

    def forward_chunk(self, hidden: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.norm(hidden)
        history = cache[:, -self.context_frames :, :]
        combined = torch.cat((history, normalized), dim=1)
        mixed = self.depthwise(combined.transpose(1, 2)).transpose(1, 2)
        mixed = self.mix(mixed)
        output = hidden + mixed * self.layer_scale

        next_history = combined[:, -self.context_frames :, :]
        missing = self.max_cache_frames - next_history.shape[1]
        if missing:
            next_history = F.pad(next_history, (0, 0, missing, 0))
        return output, next_history


class ProsodyNet(nn.Module):
    """Compact acoustic prosody encoder.

    Local embeddings are causal and streamable. The global embedding is pooled
    from the utterance/chunk and is therefore intended for offline or
    phrase-level use until M5 adds explicit streaming style state.
    """

    def __init__(self, config: ProsodyNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProsodyNetConfig()
        cfg = self.config

        self.input_projection = nn.Linear(
            cfg.acoustic_dim + cfg.pitch_dim,
            cfg.hidden_dim,
        )
        self.blocks = nn.ModuleList(
            CausalProsodyBlock(
                cfg.hidden_dim,
                cfg.kernel_size,
                dilation,
                cfg.max_cache_frames,
            )
            for dilation in cfg.dilations
        )
        self.local_norm = nn.LayerNorm(cfg.hidden_dim)
        self.local_projection = nn.Linear(cfg.hidden_dim, cfg.local_dim)
        self.local_descriptor_head = nn.Linear(
            cfg.local_dim,
            cfg.descriptor_dim,
        )

        self.global_projection = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.global_dim),
        )
        self.global_descriptor_head = nn.Linear(
            cfg.global_dim,
            cfg.descriptor_dim,
        )

    @property
    def cache_shape(self) -> tuple[int, int, int]:
        return (
            len(self.blocks),
            self.config.max_cache_frames,
            self.config.hidden_dim,
        )

    def initial_cache(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = next(self.parameters())
        return torch.zeros(
            len(self.blocks),
            batch_size,
            self.config.max_cache_frames,
            self.config.hidden_dim,
            device=reference.device if device is None else device,
            dtype=reference.dtype if dtype is None else dtype,
        )

    def forward_chunk(
        self,
        acoustic: Tensor,
        pitch: Tensor,
        cache: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_inputs(acoustic, pitch, cache)
        hidden = self.input_projection(torch.cat((acoustic, pitch), dim=-1))

        next_caches: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            hidden, next_cache = block.forward_chunk(hidden, cache[index])
            next_caches.append(next_cache)

        local = self.local_projection(self.local_norm(hidden))
        local_descriptor = self.local_descriptor_head(local)
        return local, local_descriptor, torch.stack(next_caches, dim=0)

    def forward(
        self,
        acoustic: Tensor,
        pitch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if acoustic.ndim != 3:
            raise ValueError("acoustic must have shape [batch, frames, acoustic_dim]")
        cache = self.initial_cache(
            acoustic.shape[0],
            device=acoustic.device,
            dtype=acoustic.dtype,
        )
        local, local_descriptor, _ = self.forward_chunk(acoustic, pitch, cache)

        pooled = torch.cat(
            (
                local.new_zeros(local.shape[0], self.config.hidden_dim),
                local.new_zeros(local.shape[0], self.config.hidden_dim),
            ),
            dim=-1,
        )
        hidden = self.input_projection(torch.cat((acoustic, pitch), dim=-1))
        replay_cache = self.initial_cache(
            acoustic.shape[0],
            device=acoustic.device,
            dtype=acoustic.dtype,
        )
        for index, block in enumerate(self.blocks):
            hidden, replay_cache[index] = block.forward_chunk(
                hidden,
                replay_cache[index],
            )
        mean = hidden.mean(dim=1)
        std = torch.sqrt(torch.clamp(hidden.var(dim=1, unbiased=False), min=1e-6))
        pooled = torch.cat((mean, std), dim=-1)

        global_style = self.global_projection(pooled)
        global_descriptor = self.global_descriptor_head(global_style)
        return local, global_style, local_descriptor, global_descriptor

    def _validate_inputs(
        self,
        acoustic: Tensor,
        pitch: Tensor,
        cache: Tensor,
    ) -> None:
        cfg = self.config
        if acoustic.ndim != 3 or acoustic.shape[-1] != cfg.acoustic_dim:
            raise ValueError(
                f"acoustic must have shape [batch, frames, {cfg.acoustic_dim}]"
            )
        if pitch.ndim != 3 or pitch.shape[-1] != cfg.pitch_dim:
            raise ValueError(
                f"pitch must have shape [batch, frames, {cfg.pitch_dim}]"
            )
        if acoustic.shape[:2] != pitch.shape[:2]:
            raise ValueError("acoustic and pitch must share batch/frame dimensions")

        expected = (
            len(self.blocks),
            acoustic.shape[0],
            cfg.max_cache_frames,
            cfg.hidden_dim,
        )
        if tuple(cache.shape) != expected:
            raise ValueError(f"expected cache shape {expected}, got {tuple(cache.shape)}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
