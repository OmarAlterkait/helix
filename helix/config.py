"""Detector configuration for the HELIX pipeline.

All algorithm parameters are collected into a single frozen dataclass.
No detector-specific logic in the algorithm code — everything is parameterized
through this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class DetectorConfig:
    # ── coherent removal ──
    group_size: int = 64
    beta: float = 0.15
    mask_threshold_nsigma: float = 3.0
    temporal_dilation_ticks: int = 9

    # ── wavelet sparsification ──
    wavelet: str = "coif3"
    dwt_level: int = 4
    threshold_kappa: float = 1.0
    threshold_mode: str = "hard"
    threshold_include_approx: bool = True

    # ── readout geometry ──
    num_time_steps: int = 2701
    sampling_rate_mhz: float = 2.0
    pedestals: dict[str, int] = field(default_factory=lambda: {"U": 2048, "V": 2048, "Y": 400})

    # ── intrinsic noise model (MicroBooNE-like ENC) ──
    noise_enc_x: float = 0.90
    noise_enc_y: float = 0.79
    noise_enc_z: float = 0.22

    # ── planes ──
    plane_labels: tuple[str, ...] = ("east_U", "east_V", "east_Y",
                                     "west_U", "west_V", "west_Y")

    @classmethod
    def from_toml(cls, path: str | Path) -> DetectorConfig:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(**_flatten_toml(raw))

    @classmethod
    def sbnd(cls) -> DetectorConfig:
        return cls(
            group_size=64,
            beta=0.15,
            num_time_steps=2701,
            sampling_rate_mhz=2.0,
            pedestals={"U": 2048, "V": 2048, "Y": 400},
        )

    @classmethod
    def microboone(cls) -> DetectorConfig:
        return cls(
            group_size=64,
            beta=0.15,
            num_time_steps=9594,
            sampling_rate_mhz=2.0,
            pedestals={"U": 2048, "V": 2048, "Y": 400},
            plane_labels=("U", "V", "Y"),
        )

    @classmethod
    def icarus(cls) -> DetectorConfig:
        return cls(
            group_size=64,
            beta=0.15,
            num_time_steps=4096,
            sampling_rate_mhz=2.5,
            pedestals={"U": 2048, "V": 2048, "Y": 400},
        )

    def wire_sigma_intrinsic(self, wire_lengths: Any) -> Any:
        """Compute per-wire intrinsic noise sigma from wire lengths (cm).

        σ = sqrt(x² + (y + z·L)²) where x, y, z are ENC model params.
        """
        import numpy as np
        L = np.asarray(wire_lengths, dtype=np.float32)
        return np.sqrt(self.noise_enc_x**2 + (self.noise_enc_y + self.noise_enc_z * L)**2)

    @property
    def xblock_kernel(self) -> tuple[float, float, float]:
        return (-self.beta, 1.0, -self.beta)


def _flatten_toml(raw: dict) -> dict:
    """Flatten nested TOML sections into kwargs for DetectorConfig."""
    out: dict[str, Any] = {}
    for section in ("coherent", "wavelet", "readout", "noise", "planes"):
        if section in raw:
            out.update(raw[section])
    out.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
    return out
