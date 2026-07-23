"""Spatially-local 'keep more near signal' knob = HYSTERESIS thresholding in wavelet space.

Standard VisuShrink: keep |c| > t (t = kappa*sigma_band*sqrt(2lnN)). The residual-coherent
sigma-inflation over-thresholds the SIGNAL's weak coefficients (near tracks) -> F0 loss.
Hysteresis fix: per band, SEEDS = |c| > k_seed*t ; GROW region = |c| > k_grow*t (k_grow<k_seed);
keep the connected (wire x position) components of GROW that contain a SEED. -> recovers the
signal's weak shoulders (near strong signal) while dropping ISOLATED weak coeffs (noise far from
signal). Coefficient count barely grows (only signal-adjacent coeffs added)."""
import os
import sys
import numpy as np
import pywt
from scipy import ndimage

import cc_common as cc
import smart as sm
import induction as ind

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
import common as wd  # noqa: E402
GS = cc.GROUP_SIZE
WV, LV, MD = 'coif3', 4, 'periodization'


def _mad(c):
    return np.median(np.abs(c)) / 0.6745


def sparsify_std(cleaned, kappa=1.0):
    bands = pywt.wavedec(cleaned.astype(np.float32), WV, level=LV, mode=MD, axis=-1)
    out, kept = [], 0
    for b in bands:
        s = max(_mad(b), 1e-6); t = kappa * s * np.sqrt(2 * np.log(max(b.shape[-1], 2)))
        m = np.abs(b) >= t; out.append(b * m); kept += int(m.sum())
    recon = pywt.waverec(out, WV, mode=MD, axis=-1)[..., :cleaned.shape[-1]]
    return recon.astype(np.float32), kept


def sparsify_hyst(cleaned, k_seed=1.0, k_grow=0.5):
    bands = pywt.wavedec(cleaned.astype(np.float32), WV, level=LV, mode=MD, axis=-1)
    out, kept = [], 0
    for b in bands:
        s = max(_mad(b), 1e-6); base = s * np.sqrt(2 * np.log(max(b.shape[-1], 2)))
        a = np.abs(b)
        seed = a >= k_seed * base
        grow = a >= k_grow * base
        lbl, n = ndimage.label(grow, structure=np.ones((3, 3)))   # 2D: wire x position
        keep_lab = np.unique(lbl[seed]); keep_lab = keep_lab[keep_lab > 0]
        m = np.isin(lbl, keep_lab)
        out.append(b * m); kept += int(m.sum())
    recon = pywt.waverec(out, WV, mode=MD, axis=-1)[..., :cleaned.shape[-1]]
    return recon.astype(np.float32), kept


def f0(recon, s):
    sig = np.abs(s) > 0
    return 1 - float(np.abs(recon - s)[sig].sum()) / max(float(np.abs(s)[sig].sum()), 1e-9)


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    events = list(range(0, n_ev * 9, 9))
    for p in ['U', 'V', 'Y']:
        R = {k: {'f0': [], 'kept': []} for k in
             ['std_k1.0', 'std_k0.6', 'hyst_0.5', 'hyst_0.35', 'hyst_0.2']}
        for e in events:
            s, c, i = cc.components(p, e); noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            smc = ind.smart_baseline(noisy)
            de2, _ = ind.iterate(noisy, 4, 0.7, 3.5, 15, seed='amp')
            cl = noisy - np.clip(de2, smc - 4, smc + 4)   # de2_clamp (joint step ablated to zero)
            for name, fn in [('std_k1.0', lambda: sparsify_std(cl, 1.0)),
                             ('std_k0.6', lambda: sparsify_std(cl, 0.6)),
                             ('hyst_0.5', lambda: sparsify_hyst(cl, 1.0, 0.5)),
                             ('hyst_0.35', lambda: sparsify_hyst(cl, 1.0, 0.35)),
                             ('hyst_0.2', lambda: sparsify_hyst(cl, 1.0, 0.2))]:
                rec, kept = fn(); R[name]['f0'].append(f0(rec, s)); R[name]['kept'].append(kept)
        print(f"\n  ===== {p} ({n_ev} ev) =====   {'F0':>8}  {'kept':>7}")
        for k in R:
            print(f"   {k:>10}   {np.mean(R[k]['f0']):>8.4f}  {np.mean(R[k]['kept']):>7.0f}")


if __name__ == '__main__':
    main()
