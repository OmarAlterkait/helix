"""Stronger coherent estimate using a DEEP decomposition + cross-level per-wire
signal detection.

Per-band masked-mean (smart.py) decides signal per (wire,position) by local deviation.
A deep decomposition lets us decide per WIRE per TIME using cross-level persistence: a
real pulse lights up the SAME time at many scales (cone of influence), while intrinsic
fluctuations do not. So:
  1. deep wavedec; per band, deviation of each wire from its block median, normalized by
     the per-wire fluctuation scale sigma_dev (intrinsic). z_j[w,pos].
  2. expand to time; a wire is SIGNAL at time t if |z|>zthr at >= persist levels there
     (cross-level vote); dilate in time.
  3. block common-mode per (band,pos) = mean over wires NOT signal-flagged at that time
     -> excludes whole signal wires (incl their coarse-band common-mode contribution).
  4. magnitude gate by per-band coherent scale as backup for dense-signal blocks.
This targets the coarse-band / U-A4 failure (where the per-position median is fooled by
signal that is common across the block) by using each wire's fine-scale signal signature.
"""
import os
import sys
import numpy as np
import pywt
import cc_common as cc
import smart as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('numpy')
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402
GS = cc.GROUP_SIZE


def _block_median_bcast(band, nw):
    nblk = cc.n_groups(nw)
    out = np.empty_like(band)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        out[lo:hi] = np.median(band[lo:hi], axis=0)[None, :]
    return out


def _temporal_dilate(mask, ticks):
    if ticks <= 1:
        return mask
    k = np.ones(ticks)
    return np.array([np.convolve(r.astype(np.float32), k, 'same') > 0.5 for r in mask])


def deep_removal(noisy, wavelet='sym8', level=8, mode=cc.MODE,
                 zthr=3.5, persist=2, kgate=4.0, dilate_t=11, minkeep=6):
    nw, nt = noisy.shape
    bands = pywt.wavedec(noisy.astype(np.float32), wavelet, level=level, mode=mode, axis=-1)
    nblk = cc.n_groups(nw)
    # ---- cross-level per-wire signal vote on a common time grid ----
    vote = np.zeros((nw, nt), np.int16)
    devs, sigdevs, pots = [], [], []
    for b in bands:
        Lj = b.shape[-1]
        bmed = _block_median_bcast(b, nw)
        dev = b - bmed
        sdev = max(float(np.median(np.abs(dev)) / 0.6745), 1e-6)
        pot = (np.arange(nt) * Lj) // nt
        vote += (np.abs(dev[:, pot]) / sdev > zthr).astype(np.int16)
        devs.append(dev); sigdevs.append(sdev); pots.append(pot)
    wire_sig = _temporal_dilate(vote >= persist, dilate_t)            # (nw, nt)
    # ---- common-mode per band excluding signal wires at that time ----
    est = []
    for b, pot in zip(bands, pots):
        Lj = b.shape[-1]
        bnd = np.searchsorted(pot, np.arange(Lj))
        swp = np.logical_or.reduceat(wire_sig, bnd, axis=1)          # (nw,Lj) signal mask
        M = np.zeros((nblk, Lj), np.float32)
        ngf = cc.full_groups(nw)
        if ngf:
            blk = b[:ngf * GS].reshape(ngf, GS, Lj)
            uf = ~swp[:ngf * GS].reshape(ngf, GS, Lj)
            nuf = uf.sum(1)
            mean = (blk * uf).sum(1) / np.maximum(nuf, 1)
            med = np.median(blk, axis=1)
            M[:ngf] = np.where(nuf >= minkeep, mean, med)
        if ngf * GS < nw:
            t = b[ngf * GS:]; ut = ~swp[ngf * GS:]; nuf = ut.sum(0)
            mean = (t * ut).sum(0) / np.maximum(nuf, 1)
            M[ngf] = np.where(nuf >= minkeep, mean, np.median(t, axis=0))
        # magnitude gate (backup): drop large common-modes (residual signal in dense blocks)
        sigc = max(float(np.median(np.abs(M)) / 0.6745), 1e-6)
        Mc = np.where(np.abs(M) < kgate * sigc, M, 0.0)
        est.append(sm.broadcast_blocks(Mc, nw))
    cleaned = pywt.waverec([b - e for b, e in zip(bands, est)], wavelet, mode=mode, axis=-1)[..., :nt]
    coh_hat = pywt.waverec(est, wavelet, mode=mode, axis=-1)[..., :nt]
    return cleaned.astype(np.float32), coh_hat.astype(np.float32)


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    events = list(range(0, n_ev * 13, 13))
    cfg = DetectorConfig(group_size=64)
    for p in ['Y', 'U', 'V']:
        agg = {'helix': [], 'smart_L4': [], 'deep_sym8_L8': [], 'deep_coif3_L7': []}
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            hel = np.asarray(remove_coherent(noisy, cfg))
            agg['helix'].append(sm.metrics(hel, s, c, noisy - hel))
            cl, hc = sm.smart_removal(noisy, kgate=4.0)
            agg['smart_L4'].append(sm.metrics(cl, s, c, hc))
            cl, hc = deep_removal(noisy, 'sym8', 8)
            agg['deep_sym8_L8'].append(sm.metrics(cl, s, c, hc))
            cl, hc = deep_removal(noisy, 'coif3', 7)
            agg['deep_coif3_L7'].append(sm.metrics(cl, s, c, hc))
        print(f"  plane {p}:  " + "   ".join(
            f"{k}: F0 {np.array(v).mean(0)[0]:.4f} coh {np.array(v).mean(0)[2]:.3f}"
            for k, v in agg.items()))


if __name__ == '__main__':
    main()
