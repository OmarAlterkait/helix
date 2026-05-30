"""Detector configuration for the HELIX TPC (wire) pipeline.

Algorithm parameters collected into a single frozen dataclass. Readout geometry
(time steps, plane labels, pedestals) is read from the input HDF5 file at
runtime — see helix.tpc.io.config_from_file().
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

    # ── wavelet sparsification (shared helix.core) ──
    wavelet: str = "coif3"
    dwt_level: int = 4
    dwt_mode: str = "periodization"
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
        """Per-wire intrinsic noise sigma from wire lengths (cm):
        σ = sqrt(x² + (y + z·L)²)."""
        import numpy as np
        L = np.asarray(wire_lengths, dtype=np.float32)
        return np.sqrt(self.noise_enc_x**2 + (self.noise_enc_y + self.noise_enc_z * L)**2)

    @property
    def xblock_kernel(self) -> tuple[float, float, float]:
        return (-self.beta, 1.0, -self.beta)

    def threshold_spec(self):
        """Build a helix.core ThresholdSpec from the TPC threshold fields."""
        from helix.core.wavelet import ThresholdSpec
        return ThresholdSpec(method="universal", func=self.threshold_mode,
                             scale=self.threshold_kappa,
                             include_approx=self.threshold_include_approx)
