"""Hysteresis threshold with the noise sigma estimated on LIVE rows.

:func:`helix.core.wavelet.threshold_bands` (TPC spec: universal hard threshold,
kappa * sigma_band * sqrt(2 ln N_band), per-band MAD sigma, approximation thresholded)
computes each band's MAD over ALL rows. Dead or missing channels are zero rows, so
they bias sigma low; when many are missing (SBND run 18260: 1024 of 1983 U channels of a
TPC absent in one event) sigma collapses and every coefficient of the live channels is
kept (3.3 M instead of ~35 k). :func:`threshold_bands_rows`:
  * estimates sigma on ``valid`` rows only and zeroes the others;
  * optionally per group (``sigma_scope="group"``, e.g. one sigma per front-end board)
    or per row (``"channel"``: each channel's own MAD in that band);
  * hysteresis (:mod:`helix.core.hysteresis`): the universal threshold
    kappa sigma sqrt(2 ln N_band) gives the seeds; every coefficient with |c| >= ``floor``
    sigma that is 8-connected (row x coefficient, within the band) to a seed through such
    coefficients is kept too. SBND (data, SBND MC and HELIX sim; SBND-data-helix study):
    floor 2 keeps no more noise than plain universal thresholding one step of kappa lower,
    with ~25% fewer FM tokens, and lowers the 50%-efficiency energy of isolated deposits by
    11-20%. ``floor=None``: the seeds alone (HELIX's universal threshold; with all rows
    valid and ``sigma_scope="plane"`` identical to threshold_bands with the TPC spec).

Backends: numpy (reference) and torch (``helix.tpc.threshold_rows_ops_{numpy,torch}``),
chosen by the input's type or the active backend; jax inputs run on numpy.
"""
from __future__ import annotations

import numpy as np

from helix.core import backend

_OPS = "helix.tpc.threshold_rows_ops"


def _ops_for(x):
    if backend.kind_of(x) == "torch" or backend.get_backend() == "torch":
        return backend.import_backend_module(_OPS, "torch")
    return backend.import_backend_module(_OPS, "numpy")


def _via_numpy(res):
    """jax inputs run on the numpy backend (no jax port); bands go back to jax."""
    import jax.numpy as jnp
    bands, kept, total, sig = res
    return [jnp.asarray(b) for b in bands], kept, total, sig


def _np_kw(kw):
    return {k: (np.asarray(v) if backend.kind_of(v) == "jax" else v) for k, v in kw.items()}


def band_sigma_rows(c, valid, sigma_scope="plane", groups=None):
    """Per-row sigma, shape (rows, 1)."""
    return _ops_for(c).band_sigma_rows(c, valid, sigma_scope, groups)


def threshold_bands_rows(coeffs, kappa=1.0, valid=None, sigma_scope="plane", groups=None,
                         threshold_approx=True, floor=2.0):
    """Hysteresis hard threshold per band with live-row sigma (``floor=None``: seeds only).

    Returns (bands, n_kept, n_total, sigma_per_band) like threshold_bands; sigma_per_band
    is the median of the per-row sigmas over live rows."""
    kw = dict(kappa=kappa, valid=valid, sigma_scope=sigma_scope, groups=groups, threshold_approx=threshold_approx,
              floor=floor)
    if backend.kind_of(coeffs[0]) == "jax":
        return _via_numpy(_ops_for(None).threshold_bands_rows([np.asarray(c) for c in coeffs], **_np_kw(kw)))
    return _ops_for(coeffs[0]).threshold_bands_rows(coeffs, **kw)


def threshold_planes_rows(planes):
    """Several planes at once: ``planes`` = list of (bands, kwargs of
    :func:`threshold_bands_rows`) -> list of its results. Same result as one call per plane;
    torch grows the hysteresis bands of all planes together (fewer, larger kernels)."""
    if backend.kind_of(planes[0][0][0]) == "jax":
        return [threshold_bands_rows(b, **kw) for b, kw in planes]
    return _ops_for(planes[0][0][0]).threshold_planes_rows(planes)
