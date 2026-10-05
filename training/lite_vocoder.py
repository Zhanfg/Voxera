from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class LiteVocoderConfig:
    input_mels: int = 80
    hidden_dim: int = 192
    n_fft: int = 320
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 1, 2, 4)

    def __post_init__(self) -> None:
        if min(self.input_mels, self.hidden_dim, self.n_fft) <= 0:
            raise ValueError("dimensions must be positive")
        if self.n_fft % 2:
            raise ValueError("n_fft must be even")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if not self.dilations or any(value <= 0 for value in self.dilations):
            raise ValueError("dilations must contain positive integers")

    @property
    def frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def max_cache_frames(self) -> int:
        return max(self.dilations) * (self.kernel_size - 1)


class CausalVocoderBlock(nn.Module):
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


class LiteVocoder(nn.Module):
    """Small causal mel-to-Fourier-frame network.

    The neural graph predicts log magnitude and phase only. iFFT and overlap-add
    stay outside the model in fixed DSP code, which keeps ONNX/mobile export
    simple and makes the streaming state explicit.
    """

    def __init__(self, config: LiteVocoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or LiteVocoderConfig()
        cfg = self.config

        self.input_projection = nn.Linear(cfg.input_mels, cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            CausalVocoderBlock(
                cfg.hidden_dim,
                cfg.kernel_size,
                dilation,
                cfg.max_cache_frames,
            )
            for dilation in cfg.dilations
        )
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)
        self.spectral_head = nn.Linear(
            cfg.hidden_dim,
            cfg.frequency_bins * 2,
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
        mel: Tensor,
        cache: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self._validate_inputs(mel, cache)

        hidden = self.input_projection(mel)
        next_caches: list[Tensor] = []
        for index, block in enumerate(self.blocks):
            hidden, next_cache = block.forward_chunk(hidden, cache[index])
            next_caches.append(next_cache)

        raw = self.spectral_head(self.output_norm(hidden))
        log_magnitude, phase = raw.chunk(2, dim=-1)
        log_magnitude = torch.clamp(log_magnitude, min=-20.0, max=5.0)
        return log_magnitude, phase, torch.stack(next_caches, dim=0)

    def forward(self, mel: Tensor) -> tuple[Tensor, Tensor]:
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, frames, input_mels]")
        cache = self.initial_cache(
            mel.shape[0],
            device=mel.device,
            dtype=mel.dtype,
        )
        log_magnitude, phase, _ = self.forward_chunk(mel, cache)
        return log_magnitude, phase

    def _validate_inputs(self, mel: Tensor, cache: Tensor) -> None:
        cfg = self.config
        if mel.ndim != 3:
            raise ValueError("mel must have shape [batch, frames, input_mels]")
        if mel.shape[-1] != cfg.input_mels:
            raise ValueError(
                f"expected input_mels={cfg.input_mels}, got {mel.shape[-1]}"
            )

        expected = (
            len(self.blocks),
            mel.shape[0],
            cfg.max_cache_frames,
            cfg.hidden_dim,
        )
        if tuple(cache.shape) != expected:
            raise ValueError(f"expected cache shape {expected}, got {tuple(cache.shape)}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
