"""Sample-space coherent removal with temporal gap interpolation.

Per (block,tick) the coherent = robust center (median) of the 64 wires -- correct
wherever < ~half the wires carry signal. In DENSE ticks (a track parallel to the wires
hits most of the block) the median is unreliable; but a track occupies only a LIMITED
TIME, and the coherent is SMOOTH in time, so we INTERPOLATE the coherent waveform across
the dense-tick gap from the reliable neighbors. Subtract the full (interpolated) coherent
from all wires -> removes coherent even behind dense tracks, no garbage injected.

This is the human read: the horizontal (across-wire) constancy gives the coherent where
the block is clean; time-smoothness fills it behind the track.
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


def sample_interp_removal(noisy, ksig=3.0, minrel=24, smooth=0):
    """coh[block,tick] = median over wires where reliable (>=minrel clean wires), else
    temporally interpolated. Subtract from all wires."""
    nw, nt = noisy.shape
    nblk = cc.n_groups(nw)
    coh_hat = np.empty_like(noisy)
    xs = np.arange(nt)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        blk = noisy[lo:hi]                                   # (nb, nt)
        med = np.median(blk, axis=0)                         # robust center
        sig = max(np.median(np.abs(blk - med)) / 0.6745, 1e-6)
        nclean = (np.abs(blk - med) <= ksig * sig).sum(0)    # # wires near the center
        reliable = nclean >= minrel
        coh = med.copy()
        if reliable.sum() >= 2 and (~reliable).any():
            coh[~reliable] = np.interp(xs[~reliable], xs[reliable], med[reliable])
        elif reliable.sum() < 2:
            coh[:] = 0.0
        if smooth > 1:                                       # optional light low-pass
            k = np.ones(smooth) / smooth
            coh = np.convolve(coh, k, mode='same')
        coh_hat[lo:hi] = coh[None, :]
    cleaned = noisy - coh_hat
    return cleaned.astype(np.float32), coh_hat.astype(np.float32)


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
    variants = {'interp_r32': dict(minrel=32), 'interp_r32_s11': dict(minrel=32, smooth=11),
                'interp_r32_s21': dict(minrel=32, smooth=21), 'interp_r32_s41': dict(minrel=32, smooth=41)}
    for p in ['Y', 'U', 'V']:
        agg = {'helix': [], 'smart': []}; agg.update({k: [] for k in variants})
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            hel = np.asarray(remove_coherent(noisy, cfg)); agg['helix'].append(metrics(hel, s, c, noisy - hel))
            cl, hc = sm.smart_removal(noisy, kgate=4.0); agg['smart'].append(metrics(cl, s, c, hc))
            for k, kw in variants.items():
                cl, hc = sample_interp_removal(noisy, **kw); agg[k].append(metrics(cl, s, c, hc))
        print(f"  plane {p}:")
        for k, v in agg.items():
            f0, nr, cl = np.array(v).mean(0)
            print(f"     {k:>12}  F0 {f0:.4f}  noise {nr:.3f}  coh_left {cl:.4f}")


if __name__ == '__main__':
    main()
