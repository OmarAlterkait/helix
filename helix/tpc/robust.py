"""Per-channel robust pedestal and noise: iterative k-sigma clipped mean / rms.

On integer ADC data the median and the MAD are quantized (a median pedestal has a flat
~0.29 ADC error; the MAD of codes collapses below ~0.6 ADC), so per-channel statistics
of the RAW waveform use a clipped mean / rms with the truncation corrected:
  * seed: per-row nearest-rank quartiles (median, IQR / 1.349);
  * iterate: keep |x - mu| <= k sd (and not ``mask``), mu += mean, sd = rms / c(k),
    with c(k)^2 the variance of a unit Gaussian truncated at +-k; stop when mu and sd
    move by < ``tol`` ADC;
  * rows with fewer than 10 samples left get NaN (no estimate).
Wavelet-coefficient sigmas elsewhere stay MADs: coefficients are continuous.

Backends: numpy (reference) and torch (``helix.tpc.robust_ops_{numpy,torch}``), chosen
by the input's type or the active backend.
"""
from __future__ import annotations

from helix.core import backend

_OPS = "helix.tpc.robust_ops"


def _ops_for(x):
    if backend.kind_of(x) == "torch" or backend.get_backend() == "torch":
        return backend.import_backend_module(_OPS, "torch")
    return backend.import_backend_module(_OPS, "numpy")


def clipped_stats(x, mask=None, k=3.5, n_iter=5, center=None, sd0=None, tol=1e-3):
    """Per row of ``x`` (rows, n): (mean, rms, n_used). See the module docstring."""
    return _ops_for(x).clipped_stats(x, mask=mask, k=k, n_iter=n_iter, center=center, sd0=sd0, tol=tol)
