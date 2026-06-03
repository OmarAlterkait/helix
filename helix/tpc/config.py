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
    # group_size MUST match the producer's coherent grouping (JAXTPC
    # simulation.coherent_noise.group_size / the ChannelGroupMap used at
    # injection) — it is the whole forward↔inverse "don't drift" contract.
    group_size: int = 64
    # beta is the forward adjacent-group anti-correlation coefficient (a mirror
    # of the injector's value, for provenance/assertion). It is NOT used by the
    # per-group median removal — see xblock_kernel.
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
        """Per-wire intrinsic noise sigma from wire lengths in METERS:
        σ = sqrt(x² + (y + z·L)²).

        Units match the forward model: noise_enc_z is ADC/m (MicroBooNE,
        arXiv:1705.07341; the source spectrum is calibrated at L=2.330 m), so
        ``wire_lengths`` must be in meters, not centimeters."""
        import numpy as np
        L = np.asarray(wire_lengths, dtype=np.float32)
        return np.sqrt(self.noise_enc_x**2 + (self.noise_enc_y + self.noise_enc_z * L)**2)

    @property
    def xblock_kernel(self) -> tuple[float, float, float]:
        """Forward adjacent-group coupling operator (-β, 1, -β).

        This is the *forward* coupling the injector applies to group waveforms,
        NOT its inverse. The per-group median in ``remove_coherent`` subtracts
        the shared waveform directly and is agnostic to how it was constructed,
        so this kernel is deliberately unused by removal — kept as a documented
        mirror of the forward model for provenance / model-based estimators."""
        return (-self.beta, 1.0, -self.beta)

    def threshold_spec(self):
        """Build a helix.core ThresholdSpec from the TPC threshold fields.

        TPC default = the original pre-optical suppression: per-band MAD sigma
        (colored intrinsic noise needs each band thresholded at its own level)
        and threshold the approximation band too. This yields ~60k coeffs/plane,
        vs ~930k with the optical-style single-sigma keep-approx."""
        from helix.core.wavelet import ThresholdSpec
        return ThresholdSpec(method="universal", func=self.threshold_mode,
                             scale=self.threshold_kappa, per_band_sigma=True,
                             threshold_approx=self.threshold_include_approx)
