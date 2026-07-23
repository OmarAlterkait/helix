"""Smarter coherent estimate at coif3 L4 — exploit structure the magnitude gate ignores.

Two axes:
  MASK SOURCE  how we decide which wires/positions carry signal for the common-mode:
    'pos'    : deviation from the per-position block median (current smart.py) — fooled
               in DENSE blocks where the median itself is signal.
    'sample' : a wire x time signal mask built in SAMPLE space (helix-style 2-pass:
               block-median removal -> k-sigma -> temporal dilation -> re-detect),
               robust even when most wires in a block carry signal.
  GAP HANDLING what to do at positions judged signal (where coherent is left behind):
    'gate'   : zero the coherent estimate there (current) -> abandons coherent in gaps.
    'interp' : INTERPOLATE the coherent from neighboring clean positions in the same
               band+block (coherent is smooth: correlation length ~6 positions in D4),
               i.e. gap-fill the coherent through the signal, like helix's time smoothness.

Baseline 'pos'+'gate' == smart.py magnitude gate. We test all 4 combinations + a
sample-mask masked-mean with magnitude-gate backup.
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


def _dilate_t(mask, ticks):
    if ticks <= 1:
        return mask
    k = np.ones(ticks)
    return np.array([np.convolve(r.astype(np.float32), k, 'same') > 0.5 for r in mask])


def sample_signal_mask(noisy, ksig=3.0, dilate=11, n_pass=2):
    """helix-style wire x time signal mask: block-median removal -> k-sigma -> dilate,
    re-detected on the cleaned image (catches signal the first median missed)."""
    nw, nt = noisy.shape
    nblk = cc.n_groups(nw)

    def blockmed(img):
        gm = np.empty_like(img)
        for g in range(nblk):
            lo, hi = g * GS, min((g + 1) * GS, nw)
            gm[lo:hi] = np.median(img[lo:hi], axis=0)[None, :]
        return gm

    gm = blockmed(noisy)
    resid = noisy - gm
    sigw = np.maximum(np.median(np.abs(resid), axis=1) / 0.6745, 1e-6)[:, None]
    mask = _dilate_t(np.abs(resid) > ksig * sigw, dilate)
    for _ in range(n_pass - 1):
        # masked block mean as a better coherent estimate, then re-detect
        est = np.empty_like(noisy)
        for g in range(nblk):
            lo, hi = g * GS, min((g + 1) * GS, nw)
            uf = ~mask[lo:hi]
            nuf = uf.sum(0)
            est[lo:hi] = ((noisy[lo:hi] * uf).sum(0) / np.maximum(nuf, 1))[None, :]
        cleaned = noisy - est
        mask = mask | _dilate_t(np.abs(cleaned - np.median(cleaned, axis=1, keepdims=True))
                                > ksig * sigw, dilate)
    return mask


def _pos_of_tick(nt, Lj):
    return (np.arange(nt) * Lj) // nt


def _interp_fill(M, gapmask):
    """Per block row, fill gap positions by linear interp from clean positions."""
    out = M.copy()
    L = M.shape[1]
    xs = np.arange(L)
    for g in range(M.shape[0]):
        good = ~gapmask[g]
        if good.sum() >= 2 and gapmask[g].any():
            out[g, gapmask[g]] = np.interp(xs[gapmask[g]], xs[good], M[g, good])
        elif good.sum() < 2:
            out[g] = 0.0
    return out


def estimate(bands, nw, noisy, mask_src='pos', gap='gate', kgate=4.0, ksig=3.0,
             dilate=11, minkeep=6):
    nblk = cc.n_groups(nw); nt = noisy.shape[1]
    smask = sample_signal_mask(noisy, ksig, dilate) if mask_src == 'sample' else None
    est = []
    for b in bands:
        Lj = b.shape[-1]
        if mask_src == 'sample':
            pot = _pos_of_tick(nt, Lj); bnd = np.searchsorted(pot, np.arange(Lj))
            wmask = np.logical_or.reduceat(smask, bnd, axis=1)        # (nw,Lj)
            M = np.zeros((nblk, Lj), np.float32)
            ngf = cc.full_groups(nw)
            if ngf:
                blk = b[:ngf * GS].reshape(ngf, GS, Lj)
                uf = ~wmask[:ngf * GS].reshape(ngf, GS, Lj)
                nuf = uf.sum(1)
                mean = (blk * uf).sum(1) / np.maximum(nuf, 1)
                M[:ngf] = np.where(nuf >= minkeep, mean, np.median(blk, axis=1))
            if ngf * GS < nw:
                t = b[ngf * GS:]; ut = ~wmask[ngf * GS:]; nuf = ut.sum(0)
                M[ngf] = np.where(nuf >= minkeep, (t * ut).sum(0) / np.maximum(nuf, 1),
                                  np.median(t, axis=0))
        else:
            M = sm.block_common_mode(b, nw, ksig)                    # per-position median dev
        sigc = max(float(np.median(np.abs(M)) / 0.6745), 1e-6)
        gapmask = np.abs(M) >= kgate * sigc
        if gap == 'interp':
            Mc = _interp_fill(M, gapmask)
        else:  # gate
            Mc = np.where(gapmask, 0.0, M)
        est.append(sm.broadcast_blocks(Mc, nw))
    return est


def removal(noisy, **kw):
    bands = pywt.wavedec(noisy.astype(np.float32), cc.WAVELET, level=cc.LEVEL, mode=cc.MODE, axis=-1)
    est = estimate(bands, noisy.shape[0], noisy, **kw)
    cleaned = pywt.waverec([b - e for b, e in zip(bands, est)], cc.WAVELET, mode=cc.MODE, axis=-1)[..., :noisy.shape[1]]
    coh_hat = pywt.waverec(est, cc.WAVELET, mode=cc.MODE, axis=-1)[..., :noisy.shape[1]]
    return cleaned.astype(np.float32), coh_hat.astype(np.float32)


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    events = list(range(0, n_ev * 13, 13))
    cfg = DetectorConfig(group_size=64)
    variants = {'pos+gate(smart)': dict(mask_src='pos', gap='gate'),
                'pos+interp': dict(mask_src='pos', gap='interp'),
                'sample+gate': dict(mask_src='sample', gap='gate'),
                'sample+interp': dict(mask_src='sample', gap='interp')}
    for p in ['Y', 'U', 'V']:
        agg = {'helix': []}; agg.update({k: [] for k in variants})
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            agg['helix'].append(sm.metrics(np.asarray(remove_coherent(noisy, cfg)), s, c,
                                           noisy - np.asarray(remove_coherent(noisy, cfg))))
            for k, kw in variants.items():
                cl, hc = removal(noisy, **kw)
                agg[k].append(sm.metrics(cl, s, c, hc))
        print(f"  plane {p}:")
        for k, v in agg.items():
            f0, nr, cohl = np.array(v).mean(0)
            print(f"     {k:>18}  F0 {f0:.4f}  noise {nr:.3f}  coh_left {cohl:.4f}")


if __name__ == '__main__':
    main()
