"""JAX wavelet sparsification ops (GPU, matmul DWT — pywt-exact).

Uses precomputed DWT/IDWT matrices (helix.core.dwt_matrix), so coefficients
match pywt to float32. Best for SHORT signals (e.g. TPC wire waveforms, ~4k
ticks → ~75 MB matrix). NOT for long optical chunks (a 36k² matrix is ~5 GB);
use the torch backend there.

Coefficient representation: a flat ``(n_signals, n_coeffs)`` array with the
band layout ``[cA, cD_L, …, cD_1]`` given by ``band_slices``.
"""
from __future__ import annotations

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
        _cache[key] = (jnp.asarray(Wf), jnp.asarray(Wi), slices)
    return _cache[key]


def sparsify(image, wavelet: str, level: int, mode: str, th: ThresholdSpec, sigma=None) -> SparseResult:
    x = jnp.asarray(image, dtype=jnp.float32)
    n_ticks = x.shape[-1]
    Wf, Wi, slices = _matrices(wavelet, n_ticks, level, mode)
    coeffs = x @ Wf                                  # (n_sig, n_coeffs)

    band_sigma = jnp.array([jnp.median(jnp.abs(coeffs[:, s])) / 0.6745 for s in slices])  # per-band (reporting)
    approx = slices[0]
    detail = slice(approx.stop, coeffs.shape[1])
    # per-signal noise sigma (VisuShrink): caller-supplied, else MAD of finest band
    if sigma is not None:
        nsig = jnp.asarray(sigma, dtype=jnp.float32)
    else:
        nsig = jnp.median(jnp.abs(coeffs[:, slices[-1]]), axis=1) / 0.6745   # (n_sig,)

    if th.method == "universal":
        thr = jnp.zeros_like(coeffs)
        thr = thr.at[:, approx].set(coeffs[:, approx])
        for s in slices[1:]:
            band = coeffs[:, s]
            t = (th.scale * nsig * jnp.sqrt(2.0 * jnp.log(max(s.stop - s.start, 2))))[:, None]
            a = jnp.abs(band)
            if th.func == "soft":
                kept = jnp.sign(band) * jnp.maximum(a - t, 0.0)
            elif th.func == "garrote":
                kept = jnp.where(a >= t, band - t * t / jnp.where(a == 0, 1.0, band), 0.0)
            else:
                kept = jnp.where(a >= t, band, 0.0)
            thr = thr.at[:, s].set(kept)
    else:
        det = jnp.abs(coeffs[:, detail])             # (n_sig, D)
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
        mask = mask.at[:, approx].set(True)          # keep approx untouched
        thr = jnp.where(mask, coeffs, 0.0)

    n_kept = int(jnp.count_nonzero(thr))
    n_total = int(thr.size)
    return SparseResult(coeffs=thr, n_kept=n_kept, n_total=n_total,
                        sigma_per_band=np.asarray(band_sigma),
                        wavelet=wavelet, level=level, mode=mode)


def reconstruct(coeffs, wavelet: str, level: int, mode: str, n_time: int):
    _, Wi, _ = _matrices(wavelet, n_time, level, mode)
    return coeffs @ Wi
