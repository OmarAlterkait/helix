"""Workhorse for pushing U/V induction coherent removal toward the oracle.

The lever is clean-wire DETECTION. This module separates DETECTION (build a per-(wire,tick)
signal mask) from ESTIMATION (coherent = masked-mean over clean wires + interp), so we can
plug in different detectors and measure mask quality (recall/precision vs the TRUE signal)
and end metrics (F0_rem, coh_left, F0_recon) against smart and the oracle.
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
from helix.core import wavelet as cw  # noqa: E402
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402
GS = cc.GROUP_SIZE
CFG = DetectorConfig(group_size=64)


def dilate_t(mask, ticks):
    if ticks <= 1:
        return mask
    k = np.ones(ticks)
    return np.array([np.convolve(r.astype(np.float32), k, 'same') > 0.5 for r in mask])


def smart_baseline(noisy):
    _, coh = sm.smart_removal(noisy, kgate=4.0)
    nblk = cc.n_groups(noisy.shape[0])
    return np.repeat(np.stack([coh[g * GS] for g in range(nblk)])[:, None, :], GS, 1)\
        .reshape(nblk * GS, -1)[:noisy.shape[0]]


# ----------------------------------------------------------------- DETECTORS (mask builders)
def mask_true(signal, sig_adc=2.0, dilate=11):
    return dilate_t(np.abs(signal) > sig_adc, dilate)


def mask_amp(noisy, baseline, ksig=1.5, dilate=11):
    """de detector: amplitude of residual vs baseline."""
    sigw = np.maximum(np.median(np.abs(noisy - np.median(noisy, axis=1, keepdims=True)), axis=1)
                      / 0.6745, 1e-6)[:, None]
    return dilate_t(np.abs(noisy - baseline) > ksig * sigw, dilate)


def mask_hysteresis(noisy, baseline, klo=1.0, khi=3.5, dilate=5):
    """Hysteresis (Canny-style) signal detection on the residual: strong SEEDS (>khi)
    grown along contiguous 2D regions down to a LOW threshold (>klo). Keeps weak bipolar
    track tails (recall) while rejecting isolated intrinsic spikes (precision)."""
    from scipy import ndimage
    sigw = np.maximum(np.median(np.abs(noisy - np.median(noisy, axis=1, keepdims=True)), axis=1)
                      / 0.6745, 1e-6)[:, None]
    z = np.abs(noisy - baseline) / sigw
    low = z > klo; seeds = z > khi
    lbl, n = ndimage.label(low, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    seed_labels = np.unique(lbl[seeds])
    keep[seed_labels[seed_labels > 0]] = True
    mask = keep[lbl]
    return dilate_t(mask, dilate)


# ----------------------------------------------------------------- ESTIMATION from a mask
def estimate(noisy, mask, minc=4):
    """Coherent = masked-MEAN over clean wires per (block,tick); interpolate ticks with
    < minc clean wires; median fallback for a near-fully-masked block. (median/trim reducers
    were ablated as inert and removed -- mean only.)"""
    nw, nt = noisy.shape; nblk = cc.n_groups(nw); xs = np.arange(nt)
    coh = np.empty_like(noisy)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        blk = noisy[lo:hi]; uf = ~mask[lo:hi]
        nuf = uf.sum(0)
        m = (blk * uf).sum(0) / np.maximum(nuf, 1)
        rel = nuf >= minc
        if rel.sum() >= 2 and (~rel).any():
            m[~rel] = np.interp(xs[~rel], xs[rel], m[rel])
        elif rel.sum() < 2:
            m = np.median(blk, axis=0)
        coh[lo:hi] = m[None, :]
    return coh


def final_removal(noisy, clamp=4.0, n_iter=4, klo=0.7, khi=3.5, dilate=15, seed='amp'):
    """de2_clamp: the validated opt-in induction refinement = smart baseline -> iterate
    (hysteresis/amp detect + masked-mean estimate + temporal dilation) -> clamp the estimate
    to smart's coherent +/- clamp ADC. The de3 'joint' step was ablated to ZERO gain and
    dropped here (de2_clamp == de3_clamp; see RESULTS 6m). Returns (cleaned, coh_hat)."""
    _, smc = sm.smart_removal(noisy, kgate=4.0)
    de2, _ = iterate(noisy, n_iter=n_iter, klo=klo, khi=khi, dilate=dilate, seed=seed)
    coh = np.clip(de2, smc - clamp, smc + clamp)
    return (noisy - coh).astype(np.float32), coh.astype(np.float32)


def iterate(noisy, n_iter=3, klo=1.0, khi=3.5, dilate=5, minc=4, seed='amp', base=None):
    """Joint detect+estimate: hysteresis (or plain-amplitude) mask -> masked-mean coherent
    estimate -> use as next baseline. seed='ampthr' = plain amplitude threshold (collection Y);
    anything else = hysteresis (induction U/V). base: initial baseline (default smart_baseline)."""
    coh = smart_baseline(noisy) if base is None else base
    for _ in range(n_iter):
        if seed == 'ampthr':
            mask = mask_amp(noisy, coh, ksig=1.5, dilate=dilate)   # plain amplitude threshold
        else:
            mask = mask_hysteresis(noisy, coh, klo, khi, dilate)
        coh = estimate(noisy, mask, minc)
    return coh, mask


# ----------------------------------------------------------------- metrics
def full_metrics(noisy, coh_hat, signal, coherent):
    cleaned = (noisy - coh_hat).astype(np.float32)
    sig = np.abs(signal) > 0
    f0r = 1 - float(np.abs(cleaned - signal)[sig].sum()) / max(float(np.abs(signal)[sig].sum()), 1e-9)
    coh_l = float(np.sqrt(np.mean((coh_hat - coherent) ** 2)))
    res = cw.sparsify(cleaned, wavelet=CFG.wavelet, level=CFG.dwt_level, mode=CFG.dwt_mode,
                      threshold=CFG.threshold_spec())
    recon = np.asarray(cw.reconstruct(res, noisy.shape[-1]))
    f0rec = 1 - float(np.abs(recon - signal)[sig].sum()) / max(float(np.abs(signal)[sig].sum()), 1e-9)
    return f0r, coh_l, int(res.n_kept), f0rec


def mask_quality(mask, signal, sig_adc=2.0):
    truth = np.abs(signal) > sig_adc
    tp = (mask & truth).sum(); fn = (~mask & truth).sum(); fp = (mask & ~truth).sum()
    recall = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    return float(recall), float(prec), float(mask.mean())


def diag(plane='U', events=(0, 7)):
    """Diagnose: de mask vs true mask — recall/precision and resulting estimate quality."""
    cfg = CFG
    print(f"=== {plane}: detection diagnosis ===")
    print(f"   {'method':>12} {'recall':>7} {'prec':>6} {'flag%':>6} | {'cohLeft':>7} {'F0_rem':>7} {'F0_recon':>8}")
    for e in events:
        s, c, i = cc.components(plane, e); ped = cc.PLANES[plane]['pedestal']
        noisy = wd.digitize(s + c + i, ped)
        base = smart_baseline(noisy)
        # smart reference
        sm_img, sm_coh = sm.smart_removal(noisy, kgate=4.0)
        f0r, cl, _, frec = full_metrics(noisy, sm_coh, s, c)
        print(f"   {'smart':>12} {'-':>7} {'-':>6} {'-':>6} | {cl:>7.3f} {f0r:>7.4f} {frec:>8.4f}")
        for name, mask in [('de_amp_k1.5', mask_amp(noisy, base, 1.5)),
                           ('hyst_3.5_1.0', mask_hysteresis(noisy, base, 1.0, 3.5)),
                           ('hyst_3.0_0.7', mask_hysteresis(noisy, base, 0.7, 3.0)),
                           ('hyst_4_1.5', mask_hysteresis(noisy, base, 1.5, 4.0)),
                           ('true_mask', mask_true(s))]:
            coh = estimate(noisy, mask)
            f0r, cl, _, frec = full_metrics(noisy, coh, s, c)
            rec, pr, fl = mask_quality(mask, s)
            print(f"   {name:>12} {rec:>7.3f} {pr:>6.3f} {100*fl:>5.1f}% | {cl:>7.3f} {f0r:>7.4f} {frec:>8.4f}")
        print()


if __name__ == '__main__':
    p = sys.argv[1] if len(sys.argv) > 1 else 'U'
    diag(p)
