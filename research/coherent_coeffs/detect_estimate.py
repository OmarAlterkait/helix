"""Detect-then-estimate sample-space coherent removal.

Upper bound (oracle clean-wire mask) BEATS smart on all planes -> the lever is clean-wire
DETECTION, which the naive block-median fails at in dense blocks (median locks onto signal).
Fix: use TIME-CONTINUITY to build a robust coherent baseline for DETECTION, then estimate
from the correctly-identified clean wires (low variance).

  stage 1  rough per-(block,tick) masked-mean m0 + reliability (enough clean wires);
           interp m0 over unreliable (dense) ticks -> smooth coherent BASELINE
  stage 2  signal mask = |noisy - baseline| > k*sigma (dilated) -> now correct even in
           dense ticks (baseline ~ coherent, not signal)
  stage 3  coherent est = masked-mean over the detected clean wires; interp the few
           all-signal ticks; subtract from all wires.
Optionally iterate stages 1-3.
"""
import os
import sys
import numpy as np
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


def _dilate(mask, ticks):
    if ticks <= 1:
        return mask
    k = np.ones(ticks)
    return np.array([np.convolve(r.astype(np.float32), k, 'same') > 0.5 for r in mask])


def _interp_baseline(noisy, ksig, minrel):
    """Per-block detection baseline: per-tick median, interp over dense (unreliable) ticks."""
    nw, nt = noisy.shape; nblk = cc.n_groups(nw); xs = np.arange(nt)
    base = np.empty((nblk, nt), np.float32)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw); blk = noisy[lo:hi]
        med = np.median(blk, axis=0)
        s = max(np.median(np.abs(blk - med)) / 0.6745, 1e-6)
        rel = (np.abs(blk - med) <= ksig * s).sum(0) >= minrel
        base[g] = np.interp(xs, xs[rel], med[rel]) if rel.sum() >= 2 else med
    return base


def _smart_baseline(noisy):
    """Per-block detection baseline from smart's coherent estimate (robust, time-averaged)."""
    _, coh = sm.smart_removal(noisy, kgate=4.0)
    nblk = cc.n_groups(noisy.shape[0])
    return np.stack([coh[g * GS] for g in range(nblk)])


def _wire_dilate(mask, w):
    """Grow flagged wires to +/- w neighbors (track is contiguous across wires)."""
    if w <= 0:
        return mask
    out = mask.copy()
    for s in range(1, w + 1):
        out[s:] |= mask[:-s]; out[:-s] |= mask[s:]
    return out


def de_removal(noisy, ksig=3.0, minrel=32, minc=3, dilate=11, n_iter=2, baseline='interp',
               clamp=None, wdil=0):
    """clamp (ADC): if set, the final estimate is clipped to smart_baseline +/- clamp, so
    de can only refine smart by a bounded amount (safe where detection is unreliable)."""
    nw, nt = noisy.shape; nblk = cc.n_groups(nw); xs = np.arange(nt)
    sigw = np.maximum(np.median(np.abs(noisy - np.median(noisy, axis=1, keepdims=True)), axis=1)
                      / 0.6745, 1e-6)
    base = _smart_baseline(noisy) if (baseline == 'smart' or clamp is not None) \
        else _interp_baseline(noisy, ksig, minrel)
    smart_base = base if (baseline == 'smart' or clamp is not None) else None
    coh_hat = np.empty_like(noisy)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        blk = noisy[lo:hi]; sw = sigw[lo:hi][:, None]; est = base[g]
        for _ in range(n_iter):
            mask = _dilate(np.abs(blk - est[None, :]) > ksig * sw, dilate)   # detect signal vs baseline
            mask = _wire_dilate(mask, wdil)                                  # grow across wires (track)
            uf = ~mask; nuf = uf.sum(0)
            m = (blk * uf).sum(0) / np.maximum(nuf, 1)                       # estimate from clean wires
            rel = nuf >= minc
            if rel.sum() >= 2 and (~rel).any():
                m[~rel] = np.interp(xs[~rel], xs[rel], m[rel])
            elif rel.sum() < 2:
                m = base[g].copy()
            est = m
        if clamp is not None:
            est = np.clip(est, smart_base[g] - clamp, smart_base[g] + clamp)
        coh_hat[lo:hi] = est[None, :]
    return (noisy - coh_hat).astype(np.float32), coh_hat.astype(np.float32)


def metrics(cleaned, signal, coherent, coh_hat):
    sig = np.abs(signal) > 0
    f0 = 1.0 - float(np.abs(cleaned - signal)[sig].sum()) / max(float(np.abs(signal)[sig].sum()), 1e-9)
    nrms = float(np.sqrt(np.mean((cleaned - signal)[~sig] ** 2)))
    cl = float(np.sqrt(np.mean((coh_hat - coherent) ** 2)))
    return f0, nrms, cl


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    events = list(range(0, n_ev * 17, 17))
    cfg = DetectorConfig(group_size=64)
    variants = {'de_interp': dict(baseline='interp'), 'de_smart': dict(baseline='smart')}
    for p in ['Y', 'U', 'V']:
        agg = {'helix': [], 'smart': []}; agg.update({k: [] for k in variants})
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            hel = np.asarray(remove_coherent(noisy, cfg)); agg['helix'].append(metrics(hel, s, c, noisy - hel))
            cl, hc = sm.smart_removal(noisy, kgate=4.0); agg['smart'].append(metrics(cl, s, c, hc))
            for k, kw in variants.items():
                cl, hc = de_removal(noisy, **kw); agg[k].append(metrics(cl, s, c, hc))
        print(f"  plane {p}:")
        for k, v in agg.items():
            f0, nr, cl = np.array(v).mean(0)
            print(f"     {k:>8}  F0 {f0:.4f}  noise {nr:.3f}  coh_left {cl:.4f}")


if __name__ == '__main__':
    main()
