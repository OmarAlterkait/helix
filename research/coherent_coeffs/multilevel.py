"""Multi-level (cross-scale) coherent estimate.

The magnitude gate (smart.py) decides coherent-vs-signal per coefficient by |m| vs
sigma_coh. It mis-handles two cases: (a) a coherent fluctuation that happens to be
large (>k sigma) is wrongly dropped -> coherent left behind; (b) a moderate signal
common-mode (<k sigma at a coarse level) is wrongly subtracted -> signal distorted.

Better separator using the LEVELS jointly: coherent coefficients are INDEPENDENT
across levels (different frequency bands, independent random draws), but a SIGNAL
pulse is CORRELATED across levels — it lights up every scale at the same TIME (the
cone of influence). So a common-mode is SIGNAL where it persists across >= persist_min
levels at the same time, and COHERENT otherwise. We detect signal on the block
COMMON-MODE (already aggregated over 64 wires -> coherent and signal-common-mode both
boosted vs intrinsic), then subtract the coherent everywhere it is NOT signal.

  m[band][blk,pos]        within-block masked-mean common-mode (across wires)
  z = m / sigma_coh,band  per-level normalization (handles the 10x amplitude span)
  signal if  sum_band( |z| > zthr ) >= persist_min   at that TIME
  coherent_est = m where not signal, else 0           -> subtract, IDWT
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


def _pos_of_tick(nt, Lj):
    return (np.arange(nt) * Lj) // nt


def multilevel_removal(noisy, wavelet=cc.WAVELET, level=cc.LEVEL, mode=cc.MODE,
                       ksig=3.0, zthr=2.5, persist_min=2, dilate_pos=1):
    nw, nt = noisy.shape
    bands = pywt.wavedec(noisy.astype(np.float32), wavelet, level=level, mode=mode, axis=-1)
    nblk = cc.n_groups(nw)
    Ms, sigcs, potls = [], [], []
    for b in bands:
        M = sm.block_common_mode(b, nw, ksig)               # (nblk, Lj)
        sigc = max(float(np.median(np.abs(M)) / 0.6745), 1e-6)
        Ms.append(M); sigcs.append(sigc); potls.append(_pos_of_tick(nt, b.shape[-1]))
    # cross-scale signal vote on a common time grid (per block)
    vote = np.zeros((nblk, nt), np.int16)
    for M, sigc, pot in zip(Ms, sigcs, potls):
        z = np.abs(M[:, pot]) / sigc                        # (nblk, nt) expanded to ticks
        vote += (z > zthr).astype(np.int16)
    sig_tick = vote >= persist_min                          # (nblk, nt) signal-in-time per block
    # project the time signal mask back to each band's positions, gate
    est = []
    for b, M, pot in zip(bands, Ms, potls):
        Lj = b.shape[-1]
        bnd = np.searchsorted(pot, np.arange(Lj))
        pmask = np.logical_or.reduceat(sig_tick, bnd, axis=1)   # (nblk, Lj)
        if dilate_pos:
            d = pmask.copy()
            for s in range(1, dilate_pos + 1):
                d[:, s:] |= pmask[:, :-s]; d[:, :-s] |= pmask[:, s:]
            pmask = d
        Mc = np.where(pmask, 0.0, M)                        # coherent where NOT signal
        est.append(sm.broadcast_blocks(Mc, nw))
    cleaned = pywt.waverec([b - e for b, e in zip(bands, est)], wavelet, mode=mode, axis=-1)[..., :nt]
    coh_hat = pywt.waverec(est, wavelet, mode=mode, axis=-1)[..., :nt]
    return cleaned.astype(np.float32), coh_hat.astype(np.float32)


def main():
    ptype = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    n_ev = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    events = list(range(0, n_ev * 37, 37))
    cfg = DetectorConfig(group_size=64)
    print(f"=== plane {ptype}, {n_ev} events: helix vs magnitude-gate vs cross-scale ===")
    methods = {'helix': None, 'smart_k4': ('mag', 4.0)}
    for zt in (2.0, 2.5, 3.0):
        for pm in (2, 3):
            methods[f'xscale_z{zt}_p{pm}'] = ('xs', zt, pm)
    agg = {m: [] for m in methods}
    for e in events:
        s, c, i = cc.components(ptype, e)
        noisy = wd.digitize(s + c + i, cc.PLANES[ptype]['pedestal'])
        agg['helix'].append(sm.metrics(np.asarray(remove_coherent(noisy, cfg)), s, c,
                                       noisy - np.asarray(remove_coherent(noisy, cfg))))
        cl, hc = sm.smart_removal(noisy, kgate=4.0)
        agg['smart_k4'].append(sm.metrics(cl, s, c, hc))
        for m, spec in methods.items():
            if not isinstance(spec, tuple) or spec[0] != 'xs':
                continue
            _, zt, pm = spec
            cl, hc = multilevel_removal(noisy, zthr=zt, persist_min=pm)
            agg[m].append(sm.metrics(cl, s, c, hc))
    print(f"   {'method':>18}  {'F0':>8}  {'noise':>8}  {'coh_left':>9}")
    for m in methods:
        f0, nr, cl = np.array(agg[m]).mean(0)
        print(f"   {m:>18}  {f0:>8.4f}  {nr:>8.3f}  {cl:>9.4f}")


if __name__ == '__main__':
    main()
