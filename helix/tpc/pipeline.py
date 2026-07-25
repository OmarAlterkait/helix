"""End-to-end TPC pipeline: coherent removal → wavelet sparsification.

`process_plane(image, config, removal=…)` is the single orchestrator. It selects
*where* coherent removal happens by the `removal` mode (default from config):

  'gate'      — the qualified R2 smart gate, in coefficient space:
                wavedec → coherent_gate(bands) → threshold.  No image round-trip;
                the threshold σ is measured on the gated bands (removal-then-σ).
  'multipass' — the legacy R1 image-space `remove_coherent`, then sparsify.
  'none'      — sparsify with no coherent removal.

All modes return a uniform `ProcessedPlane` (cleaned image, sparse coeffs,
reconstruction). `process_event` runs every plane; `event_coeff_event` assembles
the per-plane sparse results into a single `CoeffEvent` (the corpus atom).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from helix.core.wavelet import (
    sparsify, reconstruct, wavedec, threshold_bands, SparseResult,
)
from helix.core.provenance import BasisDescriptor, padded_length_of
from helix.core.coeff_event import CoeffEvent
from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent
from helix.tpc.coherent_gate import coherent_gate

import numpy as np


def _pad_time(image, level: int):
    """Pad the last (time) axis up to a multiple of ``2**level`` (the padded-4336
    convention: ``(-4321) % 16 = 15`` → 4336). Matches the old build's
    ``pad(g, (0, (-nt) % 16))`` so coefficients are basis-consistent."""
    x = np.asarray(image, dtype=np.float32)
    npad = (-x.shape[-1]) % (1 << level)
    if npad:
        x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, npad)])
    return x


@dataclass
class ProcessedPlane:
    cleaned: Any
    sparse: SparseResult
    reconstructed: Any
    config: DetectorConfig


def process_plane(image: Any, config: DetectorConfig, sigma_per_wire: Any | None = None,
                  *, removal: str | None = None) -> ProcessedPlane:
    """One plane: (n_wires, n_ticks) → cleaned + sparse coeffs + reconstruction.

    `removal` defaults to ``config.removal``.
    """
    mode = (removal or config.removal).lower()
    n_time = image.shape[1]

    if mode == "multipass":
        cleaned = remove_coherent(image, config, sigma_per_wire)
        sparse = sparsify(cleaned, wavelet=config.wavelet, level=config.dwt_level,
                          mode=config.dwt_mode, threshold=config.threshold_spec())
    elif mode == "gate":
        xin = _pad_time(image, config.dwt_level)             # pad to 2**level multiple (4336)
        coeffs, lev = wavedec(xin, wavelet=config.wavelet, level=config.dwt_level,
                              mode=config.dwt_mode)
        gated = coherent_gate(coeffs, group_size=config.group_size, kgate=config.gate_kgate,
                              ksig=config.gate_ksig, npass=config.gate_npass, gate_approx=True)
        out, n_kept, n_total, band_sigma = threshold_bands(gated, config.threshold_spec())
        sparse = SparseResult(coeffs=out, n_kept=n_kept, n_total=n_total,
                              sigma_per_band=band_sigma, wavelet=config.wavelet,
                              level=lev, mode=config.dwt_mode)
        cleaned = reconstruct(SparseResult(coeffs=gated, n_kept=0, n_total=0, sigma_per_band=None,
                                           wavelet=config.wavelet, level=lev, mode=config.dwt_mode),
                              n_time)                        # coherent-removed image (for inspection)
    elif mode in ("none", "off"):
        cleaned = image
        xin = _pad_time(image, config.dwt_level)
        coeffs, lev = wavedec(xin, wavelet=config.wavelet, level=config.dwt_level,
                              mode=config.dwt_mode)
        out, n_kept, n_total, band_sigma = threshold_bands(coeffs, config.threshold_spec())
        sparse = SparseResult(coeffs=out, n_kept=n_kept, n_total=n_total,
                              sigma_per_band=band_sigma, wavelet=config.wavelet,
                              level=lev, mode=config.dwt_mode)
    else:
        raise ValueError(f"unknown removal mode {mode!r} (gate|multipass|none)")

    recon = reconstruct(sparse, n_time)
    return ProcessedPlane(cleaned=cleaned, sparse=sparse, reconstructed=recon, config=config)


def process_event(planes: dict[str, Any], config: DetectorConfig,
                  sigma_per_wire: dict[str, Any] | None = None,
                  *, removal: str | None = None) -> dict[str, ProcessedPlane]:
    """Process all planes of one event → ``{label: ProcessedPlane}``."""
    results = {}
    for label, image in planes.items():
        sw = sigma_per_wire.get(label) if sigma_per_wire else None
        results[label] = process_plane(image, config, sw, removal=removal)
    return results


_PLANE_IDX = {"U": 0, "V": 1, "Y": 2}


def canonical_plane_gid(label: str) -> int:
    """``'volume_{v}_{U|V|Y}'`` → ``v*3 + {U:0,V:1,Y:2}`` (matches pimm-data)."""
    vol, p = label.rsplit("_", 1)
    return int(vol.split("_")[1]) * 3 + _PLANE_IDX[p]


def basis_from_config(config: DetectorConfig, *, band_lengths, level: int) -> BasisDescriptor:
    """Build the BasisDescriptor stamped into a CoeffEvent / shard /config.

    The padded length is derived from the actual bands (not an assumed pad rule),
    so ``band_lengths`` validate for any signal length / DWT mode.
    """
    padded = padded_length_of(band_lengths, config.wavelet, level, config.dwt_mode)
    npad = padded - config.num_time_steps
    return BasisDescriptor(
        wavelet=config.wavelet, level=level, mode=config.dwt_mode,
        n_ticks_raw=config.num_time_steps, pad=npad, band_lengths=tuple(band_lengths),
        removal=dict(kind=config.removal, kgate=config.gate_kgate, ksig=config.gate_ksig,
                     npass=config.gate_npass, group_size=config.group_size),
        threshold=dict(method="universal", func=config.threshold_mode,
                       scale=config.threshold_kappa, per_band_sigma=True,
                       threshold_approx=config.threshold_include_approx),
    )


def event_coeff_event(results: dict[int, ProcessedPlane], config: DetectorConfig,
                      *, run: str = "", source_file: str = "", event: int = -1) -> CoeffEvent:
    """Assemble ``{gid: ProcessedPlane}`` into one CoeffEvent (the corpus atom)."""
    if not results:
        raise ValueError("event_coeff_event: no planes")
    any_sparse = next(iter(results.values())).sparse
    band_lengths = [c.shape[-1] for c in any_sparse.coeffs]
    basis = basis_from_config(config, band_lengths=band_lengths, level=any_sparse.level)
    sparse_by_gid = {int(gid): pp.sparse for gid, pp in results.items()}
    return CoeffEvent.from_sparse_results(sparse_by_gid, basis=basis,
                                          run=run, source_file=source_file, event=event)
