from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Audio contract shared by reference components."""

    sample_rate: int = 16_000
    frame_ms: float = 20.0
    hop_ms: float = 10.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms must not exceed frame_ms")

    @property
    def frame_samples(self) -> int:
        return round(self.sample_rate * self.frame_ms / 1000.0)

    @property
    def hop_samples(self) -> int:
        return round(self.sample_rate * self.hop_ms / 1000.0)
