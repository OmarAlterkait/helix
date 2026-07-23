"""Push the coherent estimate: iteration, cross-block, robust multi-level variants.

All operate on the within-block common-mode m[block,pos] per band, then gate by the
per-level coherent scale (subtract small/dense = coherent, keep large/sparse = signal).
We vary HOW m is estimated and HOW the gate decides:

  mag      baseline: k-sigma masked mean across wires + magnitude gate (smart.py)
  trim     m = interquartile (trimmed) mean across wires -> robust to denser signal
  iter2    2-pass: refine the signal-wire mask on the coherent-removed residual
  xblock   gate threshold per-position from MAD across blocks (coherent is dense in
           block-space, signal sparse) + beta linear-fill of signal blocks from neighbors

Measured on removal quality (F0, coh_left) across planes.
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


def cm_masked(blk, ksig=3.0):
    med = np.median(blk, axis=0)
    r = blk - med
    sg = max(np.median(np.abs(r)) / 0.6745, 1e-6)
    uf = np.abs(r) <= ksig * sg
    nuf = uf.sum(0)
    m = (blk * uf).sum(0) / np.maximum(nuf, 1)
    return np.where(nuf > 0, m, med)


def cm_trim(blk, frac=0.30):
    """Interquartile/trimmed mean per position: drop top+bottom frac of wires."""
    n = blk.shape[0]
    lo = int(frac * n); hi = n - lo
    srt = np.sort(blk, axis=0)
    return srt[lo:hi].mean(0) if hi > lo else np.median(blk, axis=0)


def estimate(bands, nw, method='mag', kgate=4.0, ksig=3.0, n_iter=2, trim=0.30):
    nblk = cc.n_groups(nw)
    est = []
    for b in bands:
        L = b.shape[-1]
        # --- per-block common mode m (nblk,L) ---
        M = np.zeros((nblk, L), np.float32)
        for g in range(nblk):
            lo, hi = g * GS, min((g + 1) * GS, nw)
            blk = b[lo:hi]
            if method == 'trim':
                M[g] = cm_trim(blk, trim)
            elif method == 'iter2':
                m = cm_masked(blk, ksig)
                for _ in range(n_iter - 1):
                    r = blk - m                              # coherent-removed residual
                    rr = r - np.median(r, axis=0)
                    sg = max(np.median(np.abs(rr)) / 0.6745, 1e-6)
                    uf = np.abs(rr) <= ksig * sg
                    nuf = uf.sum(0)
                    m = np.where(nuf > 0, (blk * uf).sum(0) / np.maximum(nuf, 1), m)
                M[g] = m
            else:
                M[g] = cm_masked(blk, ksig)
        # --- gate (per-band scalar, or per-position for xblock) ---
        if method == 'xblock':
            sigpos = np.maximum(np.median(np.abs(M), axis=0) / 0.6745, 1e-6)   # per-position scale
            t = kgate * sigpos[None, :]
            sig_blk = np.abs(M) >= t
            Mc = np.where(sig_blk, 0.0, M)
            # beta linear fill of flagged blocks from clean neighbors (lag-1 corr ~ -0.29)
            a = -0.20
            for g in range(nblk):
                if sig_blk[g].any():
                    nb = np.zeros(L); cnt = np.zeros(L)
                    for gg in (g - 1, g + 1):
                        if 0 <= gg < nblk:
                            ok = ~sig_blk[gg]
                            nb[ok] += M[gg][ok]; cnt[ok] += 1
                    fill = np.where(cnt > 0, a * nb / np.maximum(cnt, 1), 0.0)
                    Mc[g] = np.where(sig_blk[g], fill, Mc[g])
        else:
            sigc = max(float(np.median(np.abs(M)) / 0.6745), 1e-6)
            Mc = np.where(np.abs(M) < kgate * sigc, M, 0.0)
        est.append(sm.broadcast_blocks(Mc, nw))
    return est


def removal(noisy, method, **kw):
    bands = pywt.wavedec(noisy.astype(np.float32), cc.WAVELET, level=cc.LEVEL, mode=cc.MODE, axis=-1)
    est = estimate(bands, noisy.shape[0], method=method, **kw)
    cleaned = pywt.waverec([b - e for b, e in zip(bands, est)], cc.WAVELET, mode=cc.MODE, axis=-1)[..., :noisy.shape[1]]
    coh_hat = pywt.waverec(est, cc.WAVELET, mode=cc.MODE, axis=-1)[..., :noisy.shape[1]]
    return cleaned.astype(np.float32), coh_hat.astype(np.float32)


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    planes = ['Y', 'U', 'V']
    events = list(range(0, n_ev * 13, 13))
    cfg = DetectorConfig(group_size=64)
    variants = ['mag', 'trim', 'iter2', 'xblock']
    for p in planes:
        agg = {v: [] for v in variants}
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            for v in variants:
                cl, hc = removal(noisy, v)
                agg[v].append(sm.metrics(cl, s, c, hc))
        print(f"  plane {p}:  " + "   ".join(
            f"{v}: F0 {np.array(agg[v]).mean(0)[0]:.4f} coh {np.array(agg[v]).mean(0)[2]:.3f}"
            for v in variants))


if __name__ == '__main__':
    main()
