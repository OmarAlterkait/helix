"""JAX wavelet sparsification ops (GPU, matmul DWT — pywt-exact).

Uses precomputed DWT/IDWT matrices (helix.core.dwt_matrix), so coefficients
match pywt to float32. Best for SHORT signals (e.g. TPC wire waveforms, ~4k
ticks → ~75 MB matrix). NOT for long optical chunks (a 36k² matrix is ~5 GB);
use the torch backend there.

Coefficient representation: a flat ``(n_signals, n_coeffs)`` array with the
band layout ``[cA, cD_L, …, cD_1]`` given by ``band_slices``.
"""
from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp

from helix.core.wavelet import SparseResult, ThresholdSpec
from helix.core.dwt_matrix import build_dwt_matrices

_cache: dict = {}


def _matrices(wavelet, n_ticks, level, mode):
    key = (wavelet, n_ticks, level, mode)
    if key not in _cache:
        Wf, Wi, slices = build_dwt_matrices(wavelet, n_ticks, level, mode)
        # Per-coefficient universal log-factor sqrt(2 ln N_band): each detail
        # band gets its own length; the approx band (slices[0]) stays 0 so it
        # is kept untouched (threshold 0 passes any coeff through for
        # hard/soft/garrote). Lets the universal threshold run as one
        # vectorized op instead of a per-band .at[].set() copy loop.
        n_coeffs = int(Wf.shape[1])
        lf = np.zeros(n_coeffs, dtype=np.float32)
        for s in slices[1:]:
            lf[s] = np.sqrt(2.0 * np.log(max(s.stop - s.start, 2)))
        _cache[key] = (jnp.asarray(Wf), jnp.asarray(Wi), slices, jnp.asarray(lf))
    return _cache[key]


@functools.partial(jax.jit, static_argnames=("slices_tuple", "func"))
def _universal_core(x, Wf, lf, sigma, slices_tuple, scale, func):
    """Fused VisuShrink: forward DWT → per-signal threshold → kept-count.

    Whole path compiles to one executable. ``sigma`` is the per-signal noise
    (caller-supplied, else None → MAD of the finest detail band). Returns
    (thresholded coeffs, per-band reporting sigma, kept count)."""
    coeffs = x @ Wf
    band_sigma = jnp.array([jnp.median(jnp.abs(coeffs[:, a:b])) / 0.6745
                            for (a, b) in slices_tuple])
    if sigma is None:
        fa, fb = slices_tuple[-1]
        nsig = jnp.median(jnp.abs(coeffs[:, fa:fb]), axis=1) / 0.6745
    else:
        nsig = sigma
    t = (scale * nsig)[:, None] * lf[None, :]
    a = jnp.abs(coeffs)
    if func == "soft":
        thr = jnp.sign(coeffs) * jnp.maximum(a - t, 0.0)
    elif func == "garrote":
        thr = jnp.where(a >= t, coeffs - t * t / jnp.where(coeffs == 0, 1.0, coeffs), 0.0)
    else:
        thr = jnp.where(a >= t, coeffs, 0.0)
    return thr, band_sigma, jnp.count_nonzero(thr)


def sparsify(image, wavelet: str, level: int, mode: str, th: ThresholdSpec, sigma=None) -> SparseResult:
    x = jnp.asarray(image, dtype=jnp.float32)
    n_ticks = x.shape[-1]
    Wf, Wi, slices, lf = _matrices(wavelet, n_ticks, level, mode)

    if th.method == "universal":
        slices_tuple = tuple((s.start, s.stop) for s in slices)
        sig = None if sigma is None else jnp.asarray(sigma, dtype=jnp.float32)
        thr, band_sigma, nkept = _universal_core(
            x, Wf, lf, sig, slices_tuple, float(th.scale), th.func)
        return SparseResult(coeffs=thr, n_kept=int(nkept), n_total=int(thr.size),
                            sigma_per_band=np.asarray(band_sigma),
                            wavelet=wavelet, level=level, mode=mode)

    # ---- non-universal (topk / energy): eager ----
    coeffs = x @ Wf                                  # (n_sig, n_coeffs)
    band_sigma = jnp.array([jnp.median(jnp.abs(coeffs[:, s])) / 0.6745 for s in slices])
    approx = slices[0]
    detail = slice(approx.stop, coeffs.shape[1])
    det = jnp.abs(coeffs[:, detail])                 # (n_sig, D)
    D = det.shape[1]
    if th.method == "topk":
        k = max(1, int(th.keep * D))
        tvec = jax.lax.top_k(det, k)[0][:, -1]
    else:  # energy
        srt = jnp.sort(det, axis=1)[:, ::-1]
        csum = jnp.cumsum(srt ** 2, axis=1)
        tot = jnp.maximum(csum[:, -1:], 1e-30)
        kc = jnp.clip((csum < th.energy * tot).sum(axis=1), 0, D - 1)
        tvec = jnp.take_along_axis(srt, kc[:, None], axis=1)[:, 0]
    mask = (jnp.abs(coeffs) >= tvec[:, None])
    mask = mask.at[:, approx].set(True)              # keep approx untouched
    thr = jnp.where(mask, coeffs, 0.0)
    return SparseResult(coeffs=thr, n_kept=int(jnp.count_nonzero(thr)), n_total=int(thr.size),
                        sigma_per_band=np.asarray(band_sigma),
                        wavelet=wavelet, level=level, mode=mode)


def reconstruct(coeffs, wavelet: str, level: int, mode: str, n_time: int):
    Wi = _matrices(wavelet, n_time, level, mode)[1]
    return coeffs @ Wi
