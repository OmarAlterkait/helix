"""Configuration for the optical (PMT) pipeline.

Readout/geometry fields mirror goop's ``/config`` group; wavelet fields drive
the shared helix.core sparsification.

Default method (validated on goop light_output across 100 events): per-chunk
DWT (coif3, level 10, periodization) + **VisuShrink hard thresholding** — a
per-chunk noise-relative threshold t = scale·σ·sqrt(2 ln N), with σ estimated
robustly from the chunk's finest detail (padding-independent). This is the
principled denoiser-compressor for this data; ``threshold.scale`` (κ) is the
fidelity/rate knob (κ≈1 = pure denoising; κ≈1.2/1.75/2.5 hit 1×/2×/3× the noise
floor ≈ 33×/49×/85× compression). Surviving coefficients are quantized to
``quant_bits`` for storage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from helix.core.wavelet import ThresholdSpec


@dataclass(frozen=True)
class OpticalConfig:
    # ── readout geometry (populated from the file's /config group) ──
    n_pmts_per_side: int = 81
    n_channels: int = 162
    tick_ns: float = 1.0
    pedestal: float = 0.0
    gain: float = 1.0
    n_bits: int = 15
    baseline_noise_std: float = 0.0
    sides: tuple[str, ...] = ("east", "west")

    # ── wavelet sparsification (shared core) ──
    wavelet: str = "coif3"
    dwt_level: int = 10
    dwt_mode: str = "periodization"
    threshold: ThresholdSpec = field(
        default_factory=lambda: ThresholdSpec(method="universal", func="hard", scale=1.0))
    quant_bits: int = 12        # quantize surviving coefficients for storage (near-lossless ~0.1% peak)
