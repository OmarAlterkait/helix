"""Sample-space full-subtraction coherent removal ("helix done right").

Per-tick occupancy is low (a track hits few wires per tick), so a per-(block,tick)
masked-mean over the signal-free wires recovers the coherent to ~the floor (0.2 ADC)
EVEN BEHIND SIGNAL. helix does this but subtracts (nuf/64)*est (alpha down-scaling) ->
under-removes at signal ticks. We subtract the FULL masked-mean -> remove coherent
everywhere, including behind tracks. Multi-pass mask (detect on cleaned, re-estimate
from original), temporal dilation. Compared to smart (coeff gate) and helix.
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


def _dilate(mask, ticks):
    if ticks <= 1:
        return mask
    k = np.ones(ticks)
    return np.array([np.convolve(r.astype(np.float32), k, 'same') > 0.5 for r in mask])


def _block_reduce(img, fn):
    nw = img.shape[0]; nblk = cc.n_groups(nw)
    out = np.empty_like(img)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        out[lo:hi] = fn(img[lo:hi])[None, :]
    return out


def _masked_mean_est(noisy, mask):
    """Per (block,tick) mean over UNflagged wires, broadcast to wires (full, no alpha)."""
    nw = noisy.shape[0]; nblk = cc.n_groups(nw)
    est = np.empty_like(noisy)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        uf = ~mask[lo:hi]
        nuf = uf.sum(0)
        m = (noisy[lo:hi] * uf).sum(0) / np.maximum(nuf, 1)
        med = np.median(noisy[lo:hi], axis=0)
        est[lo:hi] = np.where(nuf >= 4, m, med)[None, :]
    return est


def sample_full_removal(noisy, ksig=3.0, dilate=11, npass=3):
    gm = _block_reduce(noisy, lambda b: np.median(b, axis=0))
    resid = noisy - gm
    sigw = np.maximum(np.median(np.abs(resid), axis=1) / 0.6745, 1e-6)[:, None]
    mask = _dilate(np.abs(resid) > ksig * sigw, dilate)
    est = _masked_mean_est(noisy, mask)
    cleaned = noisy - est
    for _ in range(npass - 1):
        rc = cleaned - _block_reduce(cleaned, lambda b: np.median(b, axis=0))
        mask = mask | _dilate(np.abs(rc) > ksig * sigw, dilate)
        est = _masked_mean_est(noisy, mask)         # re-estimate from ORIGINAL
        cleaned = noisy - est
    return cleaned.astype(np.float32), est.astype(np.float32)


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
    for p in ['Y', 'U', 'V']:
        agg = {'helix': [], 'smart': [], 'sample_full': []}
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            hel = np.asarray(remove_coherent(noisy, cfg))
            agg['helix'].append(metrics(hel, s, c, noisy - hel))
            cl, hc = sm.smart_removal(noisy, kgate=4.0); agg['smart'].append(metrics(cl, s, c, hc))
            cl, hc = sample_full_removal(noisy); agg['sample_full'].append(metrics(cl, s, c, hc))
        print(f"  plane {p}:")
        for k, v in agg.items():
            f0, nr, cl = np.array(v).mean(0)
            print(f"     {k:>12}  F0 {f0:.4f}  noise {nr:.3f}  coh_left {cl:.4f}")


if __name__ == '__main__':
    main()
