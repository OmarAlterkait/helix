"""Cone-of-influence coherent removal: mask signal across TIME and LEVELS.

Motivated by per_level.py: coherent is an exact within-block common-mode at every
level, but signal becomes common-mode-like in the COARSE bands (ratio>1) where
per-wire masking fails. The only axis on which signal stays separable there is
TIME (pulses are localized). So:

  1. detect signal in SAMPLE space (rough block-median removal -> k-sigma -> dilate)
     -> per-(wire,tick) signal mask  [this is exactly helix's mask]
  2. project that tick-mask onto each level's coefficient positions via the cone of
     influence (a coeff position is masked if signal falls in its time support),
     dilated by the wavelet's support
  3. estimate the coherent common-mode per (block, level, position) as the MEAN over
     wires that are NOT signal-masked there -> exact coherent where signal-free,
     adapts per scale (fine: few wires masked; coarse: many)
  4. reconstruct estimate, subtract.

This is helix's masked-group-mean generalized to be multi-scale: the same coherent
is estimated at every level from whatever (wire,position) support is signal-free.
Compared head-to-head with helix at several decomposition depths.
"""
import os
import sys

import numpy as np
import pywt

import cc_common as cc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('numpy')
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402

GS = cc.GROUP_SIZE


# ---------------------------------------------------------------- sample-space mask
def temporal_dilate(mask, ticks):
    if ticks <= 1:
        return mask
    k = np.ones(ticks)
    out = np.empty_like(mask)
    for i in range(mask.shape[0]):
        out[i] = np.convolve(mask[i].astype(np.float32), k, mode='same') > 0.5
    return out


def signal_time_mask(noisy, ksig=3.0, dilate=11):
    """Per-(wire,tick) signal mask: rough block-median removal then k-sigma + dilate."""
    nw, nt = noisy.shape
    ng = cc.n_groups(nw)
    gm = np.zeros_like(noisy)
    for g in range(ng):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        gm[lo:hi] = np.median(noisy[lo:hi], axis=0)[None, :]
    resid = noisy - gm
    sigw = np.maximum(np.median(np.abs(resid), axis=1) / 0.6745, 1e-6)[:, None]
    return temporal_dilate(np.abs(resid) > ksig * sigw, dilate)


# ---------------------------------------------------------------- cone projection
def project_mask_to_band(tmask, Lj, dilate_pos=1):
    """tmask (nw,nt) bool -> (nw,Lj) bool: position p masked if any signal tick in
    its support; dilate by +/- dilate_pos for the wavelet's spread (cone)."""
    nw, nt = tmask.shape
    pos_of_tick = (np.arange(nt) * Lj) // nt
    bnd = np.searchsorted(pos_of_tick, np.arange(Lj))
    pm = np.logical_or.reduceat(tmask, bnd, axis=1)            # (nw,Lj)
    if dilate_pos > 0:
        out = pm.copy()
        for s in range(1, dilate_pos + 1):
            out[:, s:] |= pm[:, :-s]
            out[:, :-s] |= pm[:, s:]
        pm = out
    return pm


# ---------------------------------------------------------------- remover
def multiscale_removal(noisy, wavelet, level, mode, ksig=3.0, dilate=11,
                       dilate_pos=1, minkeep=4):
    nw, nt = noisy.shape
    tmask = signal_time_mask(noisy, ksig, dilate)
    bands = pywt.wavedec(noisy.astype(np.float32), wavelet, level=level, mode=mode, axis=-1)
    ngf = cc.full_groups(nw)
    est = []
    for b in bands:
        Lj = b.shape[-1]
        pm = project_mask_to_band(tmask, Lj, dilate_pos)       # (nw,Lj) signal mask
        e = np.zeros_like(b)
        # full blocks (vectorized)
        if ngf > 0:
            blk = b[:ngf * GS].reshape(ngf, GS, Lj)
            pmb = pm[:ngf * GS].reshape(ngf, GS, Lj)
            unflag = ~pmb
            nuf = unflag.sum(axis=1)                            # (ngf,Lj)
            mean = (blk * unflag).sum(axis=1) / np.maximum(nuf, 1)
            med = np.median(blk, axis=1)                        # fallback
            mean = np.where(nuf >= minkeep, mean, med)
            e[:ngf * GS] = np.repeat(mean[:, None, :], GS, axis=1).reshape(ngf * GS, Lj)
        # tail
        if ngf * GS < nw:
            t = b[ngf * GS:]; pmt = pm[ngf * GS:]
            unflag = ~pmt; nuf = unflag.sum(0)
            mean = (t * unflag).sum(0) / np.maximum(nuf, 1)
            med = np.median(t, axis=0)
            mean = np.where(nuf >= minkeep, mean, med)
            e[ngf * GS:] = mean[None, :]
        est.append(e)
    cleaned = pywt.waverec([b - e for b, e in zip(bands, est)], wavelet, mode=mode, axis=-1)[..., :nt]
    coh_hat = pywt.waverec(est, wavelet, mode=mode, axis=-1)[..., :nt]
    return cleaned.astype(np.float32), coh_hat.astype(np.float32)


def metrics(cleaned, signal, coherent, coh_hat):
    sig = np.abs(signal) > 0
    tc = float(np.abs(signal)[sig].sum())
    f0 = 1.0 - float(np.abs(cleaned - signal)[sig].sum()) / max(tc, 1e-9)
    nrms = float(np.sqrt(np.mean((cleaned - signal)[~sig] ** 2)))
    cl = float(np.sqrt(np.mean((coh_hat - coherent) ** 2)))
    return f0, nrms, cl


def main():
    ptype = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    n_ev = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    events = list(range(0, n_ev * 37, 37))
    cfg = DetectorConfig(group_size=64)
    configs = [('coif3', 4), ('coif3', 7), ('sym4', 9), ('sym4', 4)]

    print(f"=== plane {ptype}, {n_ev} events: helix vs cone-of-influence multiscale removal ===")
    helix_rows = []
    for e in events:
        signal, coherent, intrinsic = cc.components(ptype, e)
        noisy = wd.digitize(signal + coherent + intrinsic, cc.PLANES[ptype]['pedestal'])
        hel = np.asarray(remove_coherent(noisy, cfg))
        helix_rows.append(metrics(hel, signal, coherent, noisy - hel))
    f0, nr, cl = np.array(helix_rows).mean(0)
    print(f"   {'helix (sample-space)':>26}  F0 {f0:.4f}  noise {nr:.3f}  coh_left {cl:.4f}")

    for wav, lev in configs:
        rows = []
        for e in events:
            signal, coherent, intrinsic = cc.components(ptype, e)
            noisy = wd.digitize(signal + coherent + intrinsic, cc.PLANES[ptype]['pedestal'])
            cl_img, hc = multiscale_removal(noisy, wav, lev, cc.MODE)
            rows.append(metrics(cl_img, signal, coherent, hc))
        f0, nr, cl = np.array(rows).mean(0)
        print(f"   {f'cone {wav} L{lev}':>26}  F0 {f0:.4f}  noise {nr:.3f}  coh_left {cl:.4f}")


if __name__ == '__main__':
    main()
