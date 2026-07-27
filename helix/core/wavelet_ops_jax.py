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
        lf = np.zeros(n_coeffs, dtype=np.float32)         # approx=0 -> kept untouched
        for s in slices[1:]:
            lf[s] = np.sqrt(2.0 * np.log(max(s.stop - s.start, 2)))
        lf_all = lf.copy()                                # also threshold the approx band
        a0 = slices[0]
        lf_all[a0] = np.sqrt(2.0 * np.log(max(a0.stop - a0.start, 2)))
        _cache[key] = (jnp.asarray(Wf), jnp.asarray(Wi), slices,
                       jnp.asarray(lf), jnp.asarray(lf_all))
    return _cache[key]


@functools.partial(jax.jit, static_argnames=("slices_tuple", "func", "per_band"))
def _universal_core(x, Wf, lf, sigma, slices_tuple, scale, func, per_band):
    """Fused VisuShrink: forward DWT → threshold → kept-count.

    ``per_band`` False: one per-signal sigma (caller-supplied, else MAD of finest
    detail band) for all bands — optical. True: per-band MAD sigma broadcast to
    each band's coeffs — TPC (colored noise). ``lf`` selects keep-approx vs
    threshold-approx. Returns (thresholded coeffs, per-band reporting sigma, count)."""
    coeffs = x @ Wf
    band_sigma = jnp.array([jnp.median(jnp.abs(coeffs[:, a:b])) / 0.6745
                            for (a, b) in slices_tuple])
    if per_band:
        sig_vec = jnp.concatenate([jnp.full(b - a, band_sigma[k])
                                   for k, (a, b) in enumerate(slices_tuple)])
        t = scale * sig_vec[None, :] * lf[None, :]
    else:
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
    Wf, Wi, slices, lf, lf_all = _matrices(wavelet, n_ticks, level, mode)

    if th.method == "universal":
        slices_tuple = tuple((s.start, s.stop) for s in slices)
        sig = None if sigma is None else jnp.asarray(sigma, dtype=jnp.float32)
        lf_use = lf_all if th.threshold_approx else lf
        thr, band_sigma, nkept = _universal_core(
            x, Wf, lf_use, sig, slices_tuple, float(th.scale), th.func, bool(th.per_band_sigma))
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
    """Inverse matmul DWT. Accepts a band list (the ``wavedec`` seam) or the flat
    ``sparsify`` layout.

    The inverse matrix must be built for the length the coefficients actually
    describe — which is the PADDED signal length, not ``n_time``. (For
    periodization ``sum(band_lengths) != n_input`` in general: coif3 L4 on 4321
    ticks yields 4325 coefficients, on 4336 yields 4336.) We recover it the same
    way ``provenance.padded_length_of`` does — via a zero ``waverec`` — then crop
    to ``n_time``, matching the numpy backend's ``waverec(...)[..., :n_time]``.
    """
    if isinstance(coeffs, (list, tuple)):
        import pywt
        band_lengths = [int(np.asarray(c).shape[-1]) for c in coeffs]
        rec_len = int(pywt.waverec([np.zeros(L, np.float32) for L in band_lengths],
                                   wavelet, mode=mode).shape[-1])
        flat = jnp.concatenate([jnp.asarray(c) for c in coeffs], axis=-1)
    else:                                          # legacy flat layout from sparsify
        flat = jnp.asarray(coeffs)
        rec_len = n_time
    Wi = _matrices(wavelet, rec_len, level, mode)[1]
    return (flat @ Wi)[..., :n_time]


# ---- the transform/threshold seam (mirrors wavelet_ops_numpy) --------------
#
# ``sparsify`` above is the fused fast path. These two expose the seam a
# coefficient-space step (the coherent gate) needs to sit in, and return a BAND
# LIST so the contract matches the numpy backend (CoeffEvent.from_sparse_results
# and coherent_gate both consume band lists).

def _eff_level(n_ticks: int, wavelet: str, level: int) -> int:
    import pywt
    return min(level, pywt.dwt_max_level(n_ticks, pywt.Wavelet(wavelet).dec_len))


def wavedec(image, wavelet: str, level: int, mode: str):
    """Forward matmul DWT -> ``([cA, cD_L, …, cD_1], effective_level)`` (GPU)."""
    x = jnp.asarray(image, dtype=jnp.float32)
    lev = _eff_level(x.shape[-1], wavelet, level)
    Wf, _, slices, _, _ = _matrices(wavelet, x.shape[-1], lev, mode)
    flat = x @ Wf
    return [flat[..., s] for s in slices], lev


def _mad_sigma_j(c):
    return jnp.median(jnp.abs(c)) / 0.6745


def threshold_bands(coeffs, th: ThresholdSpec, sigma=None):
    """Threshold a band list -> ``(out_bands, n_kept, n_total, band_sigma)`` (GPU).

    Same estimator as the numpy backend: per-band MAD sigma measured on the
    (already gated) coefficients, universal threshold ``scale*sigma*sqrt(2 ln N)``.
    """
    bands = [jnp.asarray(c, dtype=jnp.float32) for c in coeffs]
    band_sigma = jnp.stack([_mad_sigma_j(c) for c in bands])

    if th.method == "universal":
        if sigma is None:
            nsig = jnp.median(jnp.abs(bands[-1]), axis=-1) / 0.6745
        else:
            nsig = jnp.asarray(sigma, dtype=jnp.float32)
        out = []
        for i, c in enumerate(bands):
            if i == 0 and not th.threshold_approx:
                out.append(c)
                continue
            lf = float(np.sqrt(2.0 * np.log(max(c.shape[-1], 2))))
            t = (th.scale * band_sigma[i] * lf) if th.per_band_sigma \
                else (th.scale * nsig[..., None] * lf)
            a = jnp.abs(c)
            if th.func == "soft":
                out.append(jnp.sign(c) * jnp.maximum(a - t, 0.0))
            elif th.func == "garrote":
                out.append(jnp.where(a >= t, c - t * t / jnp.where(c == 0, 1.0, c), 0.0))
            else:
                out.append(jnp.where(a >= t, c, 0.0))
    else:                                           # topk / energy (approx kept)
        det = jnp.concatenate(bands[1:], axis=-1)
        a = jnp.abs(det)
        D = a.shape[-1]
        if th.method == "topk":
            k = max(1, int(th.keep * D))
            tvec = jax.lax.top_k(a, k)[0][..., -1:]
        else:
            srt = jnp.sort(a, axis=-1)[..., ::-1]
            csum = jnp.cumsum(srt ** 2, axis=-1)
            tot = jnp.maximum(csum[..., -1:], 1e-30)
            kc = jnp.clip((csum < th.energy * tot).sum(axis=-1), 0, D - 1)
            tvec = jnp.take_along_axis(srt, kc[..., None], axis=-1)
        out = [bands[0]] + [jnp.where(jnp.abs(c) >= tvec, c, 0.0) for c in bands[1:]]

    # ONE device->host sync for the whole band list, not one per band.
    n_kept = int(jnp.sum(jnp.stack([jnp.count_nonzero(c) for c in out])))
    n_total = int(sum(int(c.size) for c in out))       # shapes only, no sync
    return out, n_kept, n_total, np.asarray(band_sigma)
