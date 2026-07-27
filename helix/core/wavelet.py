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


class FlatBands:
    """A band list backed by ONE flat ``(..., sum(lens))`` array.

    The jax path keeps coefficients flat end to end: ``wavedec`` produces one
    array, the gate and threshold consume and return it untouched, and only the
    final sparse extraction leaves the device. Indexing yields cheap slice views,
    so anything written against the band-list contract still works — but the jax
    ops detect this type and operate on ``.flat`` directly, avoiding the
    concatenate/split round trip that a real list forces (four full-array copies
    per plane, which cost more than the dispatches it saved).
    """
    __slots__ = ("flat", "lens", "offs")

    def __init__(self, flat, lens):
        self.flat = flat
        self.lens = tuple(int(x) for x in lens)
        o, acc = [0], 0
        for L in self.lens:
            acc += L; o.append(acc)
        self.offs = tuple(o)

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        return self.flat[..., self.offs[i]:self.offs[i + 1]]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def like(self, flat):
        return FlatBands(flat, self.lens)


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


def wavedec(image: Any, *, wavelet: str = "coif3", level: int = 4,
            mode: str = "periodization"):
    """Forward DWT → ``([cA, cD_L, …, cD_1], effective_level)`` (active backend).

    The transform half of the pipeline, exposed so a coefficient-space step (the
    coherent gate) can run between the DWT and thresholding. Numpy backend.
    """
    return _backend.ops(_OPS).wavedec(image, wavelet, level, mode)


def threshold_bands(coeffs, threshold: ThresholdSpec | None = None, sigma: Any = None):
    """Threshold a band list → ``(out_bands, n_kept, n_total, band_sigma)`` (active backend)."""
    return _backend.ops(_OPS).threshold_bands(coeffs, threshold or ThresholdSpec(), sigma)
