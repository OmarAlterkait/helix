"""Detector configuration for the HELIX pipeline.

Algorithm parameters are collected into a single frozen dataclass.
Readout geometry (time steps, plane labels, pedestals) is read from
the input HDF5 file at runtime — see helix.io.config_from_file().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DetectorConfig:
    # ── coherent removal ──
    group_size: int = 64
    beta: float = 0.15
    mask_threshold_nsigma: float = 3.0
    temporal_dilation_ticks: int = 11
    n_passes: int = 3

    # ── wavelet sparsification ──
    wavelet: str = "coif3"
    dwt_level: int = 4
    threshold_kappa: float = 1.0
    threshold_mode: str = "hard"
    threshold_include_approx: bool = True

    # ── readout geometry (populated from input file) ──
    num_time_steps: int = 2701
    sampling_rate_mhz: float = 2.0
    pedestals: dict[str, int] = field(default_factory=lambda: {"U": 2048, "V": 2048, "Y": 400})
    plane_labels: tuple[str, ...] = ()

    # ── intrinsic noise model (MicroBooNE-like ENC) ──
    noise_enc_x: float = 0.90
    noise_enc_y: float = 0.79
    noise_enc_z: float = 0.22

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
