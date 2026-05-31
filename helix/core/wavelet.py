"""Detector-agnostic wavelet sparsification — backend-dispatched.

This is the piece shared by the TPC (wire) and optical (PMT) pipelines: a
per-signal 1-D DWT, coefficient thresholding, and inverse DWT. The actual
array math lives in ``wavelet_ops_{numpy,jax,torch}.py``; this module only
holds the backend-independent data structures and the dispatch entry points.

Threshold strategies (validated by the optical sweep + the prior handoff):
  - universal : Donoho-Johnstone  t = scale·sigma_band·sqrt(2 ln N)
                func='hard' (best charge/area preservation) or 'garrote'
  - topk      : keep the top ``keep`` fraction of detail coeffs per signal
                (best compression-vs-fidelity front on the optical data)
  - energy    : keep the smallest set of detail coeffs holding ``energy``
                fraction of the detail energy, per signal
The approximation band is kept untouched when ``include_approx`` is True.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from helix.core import backend as _backend

_OPS = "helix.core.wavelet_ops"


@dataclass(frozen=True)
class ThresholdSpec:
    method: str = "universal"   # 'universal' | 'topk' | 'energy'
    func: str = "hard"          # 'hard' | 'garrote'  (universal only)
    scale: float = 1.0          # universal threshold multiplier (kappa)
    keep: float = 0.01          # topk: fraction of detail coeffs to keep
    energy: float = 0.999       # energy: fraction of detail energy to keep
    include_approx: bool = True  # (legacy/no-op) approx kept unless threshold_approx
    per_band_sigma: bool = False    # universal: per-band MAD sigma (else single finest/caller sigma)
    threshold_approx: bool = False  # universal: also threshold the approx band (else keep it)


@dataclass
class SparseResult:
    """Thresholded DWT coefficients + bookkeeping.

    ``coeffs`` is backend-native: a list ``[cA, cD_L, …, cD_1]`` on the numpy
    backend, a flat ``(n_signals, n_coeffs)`` array on jax/torch (matmul DWT).
    """
    coeffs: Any
    n_kept: int
    n_total: int
    sigma_per_band: Any
    wavelet: str
    level: int
    mode: str

    @property
    def sparsity(self) -> float:
        return 1.0 - self.n_kept / max(self.n_total, 1)

    @property
    def compression(self) -> float:
        return self.n_total / max(self.n_kept, 1)


def sparsify(
    image: Any,
    *,
    wavelet: str = "coif3",
    level: int = 4,
    mode: str = "periodization",
    threshold: ThresholdSpec | None = None,
    sigma: Any = None,
) -> SparseResult:
    """Per-signal DWT -> threshold -> sparse coefficients (active backend).

    ``sigma`` (optional, per-signal noise level) is used by the 'universal'
    (VisuShrink) threshold. If None, it is estimated as the MAD of the finest
    detail band. Supply it when padding would corrupt that estimate (optical).
    """
    return _backend.ops(_OPS).sparsify(
        image, wavelet, level, mode, threshold or ThresholdSpec(), sigma)


def reconstruct(result: SparseResult, n_time: int) -> Any:
    """Inverse DWT from a :class:`SparseResult` (active backend)."""
    return _backend.ops(_OPS).reconstruct(
        result.coeffs, result.wavelet, result.level, result.mode, n_time)
