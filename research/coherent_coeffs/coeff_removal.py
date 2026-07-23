"""Coefficient-space coherent removal — exploiting within-block identity.

Idea (the payoff of the structure study): within a 64-wire block the coherent
wavelet coefficients are IDENTICAL across wires, signal is SPARSE (few wires per
coeff position), and intrinsic is zero-median per wire. So the per-coefficient
MEDIAN across a block's wires recovers the coherent coefficient almost exactly:

    coh_hat[g, band, pos] = median_{wire in block g} coeff[wire, band, pos]

Subtract from every wire, inverse-DWT -> cleaned image. This is helix's
sample-space group-median idea moved into coefficient space, where signal is
sparser (so the median is cleaner) and the coherent estimate is exact-in-the-limit.

Compared against:
  raw          no removal
  helix        production sample-space multi-pass removal (helix.tpc.remove_coherent)
  coeff_all    coeff-space block-median, all bands
  coeff_coarse coeff-space block-median, coarse bands only (A4..D3); fine bands left

Metrics on noisy = signal + coherent + intrinsic:
  F0           fidelity on signal pixels (higher better)
  noise_rms    RMS off-signal vs clean signal (lower better)
  coh_left     RMS of leftover coherent = recon(coh_hat) - coherent (lower better)
"""
import os
import sys

import numpy as np
import pywt

import cc_common as cc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # repo root (helix pkg)
import common as wd  # noqa: E402

from helix.core import backend as _be  # noqa: E402
_be.set_backend('numpy')
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402

GS = cc.GROUP_SIZE


def block_median_estimate(bands, nw, coarse_only=False):
    """Per-band, per-coeff block-median across wires -> broadcast back to wires.

    Returns a list of (nw, len_band) coherent estimates (same band layout)."""
    labels = cc.band_labels()
    ng = cc.n_groups(nw)
    est = []
    for bi, (b, lab) in enumerate(zip(bands, labels)):
        if coarse_only and lab in ('D1', 'D2'):
            est.append(np.zeros_like(b)); continue
        e = np.zeros_like(b)
        for g in range(ng):
            lo, hi = g * GS, min((g + 1) * GS, nw)
            med = np.median(b[lo:hi], axis=0)          # (len_band,)
            e[lo:hi] = med[None, :]
        est.append(e)
    return est


def block_masked_estimate(bands, nw, ksig=3.0, est_bands=None):
    """Coefficient-space analog of helix removal: per coeff, take the block MEDIAN
    (robust common-mode), flag wires deviating > ksig*MAD (signal coeffs), then
    re-estimate coherent as the MEAN over unflagged wires. Excludes signal from
    the common-mode estimate -> clean coherent recovery without distorting signal.

    est_bands: set of band indices to estimate (others -> zero = left untouched).
    Default None = all bands. Use {1..L} to skip the signal-dominated approx band."""
    labels = cc.band_labels()
    est = []
    for bi, (b, lab) in enumerate(zip(bands, labels)):
        if est_bands is not None and bi not in est_bands:
            est.append(np.zeros_like(b)); continue
        e = np.zeros_like(b)
        ngf = cc.full_groups(nw)
        L = b.shape[-1]
        # vectorized over full blocks
        if ngf > 0:
            blk = b[:ngf * GS].reshape(ngf, GS, L)
            med = np.median(blk, axis=1, keepdims=True)              # (ngf,1,L)
            resid = blk - med
            sig = (np.median(np.abs(resid), axis=(1, 2)) / 0.6745)   # (ngf,)
            sig = np.maximum(sig, 1e-6)[:, None, None]
            unflag = np.abs(resid) <= ksig * sig                     # keep coherent+noise wires
            nuf = unflag.sum(axis=1)                                 # (ngf,L)
            mean = (blk * unflag).sum(axis=1) / np.maximum(nuf, 1)   # (ngf,L)
            mean = np.where(nuf > 0, mean, med[:, 0, :])             # fallback to median
            e[:ngf * GS] = np.repeat(mean[:, None, :], GS, axis=1).reshape(ngf * GS, L)
        # tail (partial) block
        if ngf * GS < nw:
            t = b[ngf * GS:]
            med = np.median(t, axis=0)
            resid = t - med
            sig = max(float(np.median(np.abs(resid)) / 0.6745), 1e-6)
            unflag = np.abs(resid) <= ksig * sig
            nuf = unflag.sum(axis=0)
            mean = (t * unflag).sum(axis=0) / np.maximum(nuf, 1)
            mean = np.where(nuf > 0, mean, med)
            e[ngf * GS:] = mean[None, :]
        est.append(e)
    return est


def _reconstruct_pair(noisy, bands, est):
    clean_bands = [b - e for b, e in zip(bands, est)]
    cleaned = pywt.waverec(clean_bands, cc.WAVELET, mode=cc.MODE, axis=-1)[..., :noisy.shape[1]]
    coh_hat = pywt.waverec(est, cc.WAVELET, mode=cc.MODE, axis=-1)[..., :noisy.shape[1]]
    return cleaned.astype(np.float32), coh_hat.astype(np.float32)


def coeff_space_removal(noisy, est_bands=None, masked=True, ksig=3.0):
    nw = noisy.shape[0]
    bands = cc.dwt_bands(noisy)
    if masked:
        est = block_masked_estimate(bands, nw, ksig=ksig, est_bands=est_bands)
    else:
        est = block_median_estimate(bands, nw)
    return _reconstruct_pair(noisy, bands, est)


def metrics(cleaned, signal, coherent, coh_hat=None):
    sig = np.abs(signal) > 0
    tc = float(np.abs(signal)[sig].sum())
    f0 = 1.0 - float(np.abs(cleaned - signal)[sig].sum()) / max(tc, 1e-9)
    off = ~sig
    nrms = float(np.sqrt(np.mean((cleaned - signal)[off] ** 2)))
    cl = float(np.sqrt(np.mean((coh_hat - coherent) ** 2))) if coh_hat is not None else np.nan
    return f0, nrms, cl


def main():
    ptype = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    n_ev = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    events = list(range(0, n_ev * 37, 37))
    cfg = DetectorConfig(group_size=64)

    print(f"=== plane {ptype}, {n_ev} events, {cc.WAVELET} L{cc.LEVEL} {cc.MODE} ===")
    print(f"   coherent removal: noisy = signal + coherent + intrinsic\n")
    L = cc.LEVEL
    detail = set(range(1, L + 1))                 # skip band 0 (approx, signal-dominated)
    order = ('raw', 'helix', 'coeff_all_masked', 'coeff_detail', 'helix+coeff_detail')
    agg = {k: [] for k in order}
    for e in events:
        signal, coherent, intrinsic = cc.components(ptype, e)
        ped = cc.PLANES[ptype]['pedestal']
        noisy = wd.digitize(signal + coherent + intrinsic, ped)

        # raw (no removal)
        agg['raw'].append(metrics(noisy, signal, coherent, coh_hat=np.zeros_like(coherent)))
        # helix sample-space removal
        hel = np.asarray(remove_coherent(noisy, cfg))
        agg['helix'].append(metrics(hel, signal, coherent, coh_hat=(noisy - hel)))
        # coeff-space masked mean, ALL bands (incl approx -> fails on signal blocks)
        ca, hca = coeff_space_removal(noisy, est_bands=None, masked=True)
        agg['coeff_all_masked'].append(metrics(ca, signal, coherent, hca))
        # coeff-space masked mean, DETAIL bands only (approx left untouched)
        cd, hcd = coeff_space_removal(noisy, est_bands=detail, masked=True)
        agg['coeff_detail'].append(metrics(cd, signal, coherent, hcd))
        # hybrid: helix (handles low-freq/approx) THEN coeff-space detail cleanup
        cd2, _ = coeff_space_removal(hel, est_bands=detail, masked=True)
        agg['helix+coeff_detail'].append(metrics(cd2, signal, coherent, coh_hat=(noisy - cd2)))

    print("WITH SIGNAL (realistic):")
    print(f"   {'method':>20}  {'F0':>8}  {'noise_rms':>10}  {'coh_left_rms':>12}")
    print(f"   {'-'*20}  {'-'*8}  {'-'*10}  {'-'*12}")
    for k in order:
        a = np.array(agg[k])
        f0, nr, cl = a.mean(0)
        print(f"   {k:>20}  {f0:>8.4f}  {nr:>10.3f}  {cl:>12.4f}")
    print(f"   (coherent input RMS ~ {coherent.std():.3f} ADC; coh_left_rms=0 is perfect removal)")

    # ---- noise-only regime: signal=0 isolates the estimator's intrinsic accuracy ----
    print("\nNOISE-ONLY (signal=0) — isolates estimator accuracy (no signal common-mode):")
    n_order = ('raw', 'helix', 'coeff_all_masked')
    nagg = {k: [] for k in n_order}
    for e in events:
        _, coherent, intrinsic = cc.components(ptype, e)
        ped = cc.PLANES[ptype]['pedestal']
        zero = np.zeros_like(coherent)
        noisy = wd.digitize(coherent + intrinsic, ped)
        nagg['raw'].append(metrics(noisy, zero, coherent, coh_hat=zero))
        hel = np.asarray(remove_coherent(noisy, cfg))
        nagg['helix'].append(metrics(hel, zero, coherent, coh_hat=(noisy - hel)))
        ca, hca = coeff_space_removal(noisy, est_bands=None, masked=True)
        nagg['coeff_all_masked'].append(metrics(ca, zero, coherent, hca))
    print(f"   {'method':>20}  {'noise_rms':>10}  {'coh_left_rms':>12}")
    print(f"   {'-'*20}  {'-'*10}  {'-'*12}")
    for k in n_order:
        _, nr, cl = np.array(nagg[k]).mean(0)
        print(f"   {k:>20}  {nr:>10.3f}  {cl:>12.4f}")


if __name__ == '__main__':
    main()
