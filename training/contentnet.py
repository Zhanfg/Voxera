from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ContentNetConfig:
    """Compact causal content encoder intended for teacher distillation."""

    input_dim: int = 80
    hidden_dim: int = 192
    output_dim: int = 256
    kernel_size: int = 3
    dilations: tuple[int, ...] = (
        1,
        2,
        4,
        8,
        16,
        32,
        1,
        2,
        4,
        8,
        16,
        32,
    )

    def __post_init__(self) -> None:
        if min(self.input_dim, self.hidden_dim, self.output_dim) <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must contain positive integers")

    @property
    def max_cache_frames(self) -> int:
        return max(self.dilations) * (self.kernel_size - 1)


class CausalDepthwiseBlock(nn.Module):
    """Depthwise temporal context + channel MLP with a bounded streaming cache."""

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


class CausalContentNet(nn.Module):
    """~1.86M parameter streaming content encoder.

    Input:
        [batch, frames, 80] log-mel features at 10 ms cadence.

    Output:
        [batch, frames, 256] dense content embeddings. Voxera's runtime cadence
        selector takes every fourth frame, producing a 40 ms VC feature cadence.

    The streaming cache has fixed shape:
        [layers, batch, max_cache_frames, hidden_dim]

    This fixed-shape state is deliberate: it maps cleanly to ONNX and mobile
    runtimes without Python objects or variable-length KV caches.
    """

    def __init__(self, config: ContentNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContentNetConfig()
        max_cache = self.config.max_cache_frames

        self.input_projection = nn.Linear(self.config.input_dim, self.config.hidden_dim)
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(
                self.config.hidden_dim,
                self.config.kernel_size,
                dilation,
                max_cache,
            )
            for dilation in self.config.dilations
        )
        self.output_norm = nn.LayerNorm(self.config.hidden_dim)
        self.output_projection = nn.Linear(
            self.config.hidden_dim,
            self.config.output_dim,
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

    def forward_chunk(self, features: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_inputs(features, cache)
        hidden = self.input_projection(features)
        next_caches: list[Tensor] = []

        for index, block in enumerate(self.blocks):
            hidden, next_cache = block.forward_chunk(hidden, cache[index])
            next_caches.append(next_cache)

        output = self.output_projection(self.output_norm(hidden))
        return output, torch.stack(next_caches, dim=0)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, frames, input_dim]")
        cache = self.initial_cache(
            features.shape[0],
            device=features.device,
            dtype=features.dtype,
        )
        output, _ = self.forward_chunk(features, cache)
        return output

    def _validate_inputs(self, features: Tensor, cache: Tensor) -> None:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, frames, input_dim]")
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected input_dim={self.config.input_dim}, got {features.shape[-1]}"
            )

        expected = (
            len(self.blocks),
            features.shape[0],
            self.config.max_cache_frames,
            self.config.hidden_dim,
        )
        if tuple(cache.shape) != expected:
            raise ValueError(f"expected cache shape {expected}, got {tuple(cache.shape)}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
