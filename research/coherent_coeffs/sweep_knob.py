"""One-at-a-time knob sweep for de2_clamp, 4 metrics per plane (F0, kept, nz_in, nz_out).
Usage: sweep_knob.py <knob1,knob2,...> [n_ev]. Each knob varied; others at default.
Knobs: klo khi dilate n_iter reducer minc clamp detector baseline kgate."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import sys
import numpy as np
import cc_common as cc
import smart as sm
import induction as ind
import detect_estimate as de

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('jax')
from helix.core import wavelet as cw  # noqa: E402
from helix.tpc.config import DetectorConfig  # noqa: E402
CFG = DetectorConfig(group_size=64); GS = 64
DEF = dict(klo=0.7, khi=3.5, dilate=15, n_iter=4, minc=4,
           detector='amp', clamp=4.0, kgate=4.0, baseline='smart')
SWEEP = {'klo': [0.5, 0.7, 0.9, 1.1], 'khi': [2.5, 3.5, 4.5], 'dilate': [1, 5, 11, 15, 21, 31],
         'n_iter': [1, 2, 4, 6], 'minc': [2, 4, 8, 16],
         'clamp': [2.0, 3.0, 4.0, 6.0, 8.0, 1e9], 'detector': ['amp', 'ampthr'],
         'baseline': ['smart', 'interp', 'median'], 'kgate': [3.0, 4.0, 5.0]}


def smc_k(noisy, kgate):
    _, coh = sm.smart_removal(noisy, kgate=kgate)
    nblk = cc.n_groups(noisy.shape[0])
    return np.repeat(np.stack([coh[g * GS] for g in range(nblk)])[:, None, :], GS, 1).reshape(nblk * GS, -1)[:noisy.shape[0]]


def median_base(noisy):
    nblk = cc.n_groups(noisy.shape[0]); b = np.empty_like(noisy)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, noisy.shape[0]); b[lo:hi] = np.median(noisy[lo:hi], axis=0)[None, :]
    return b


def build(noisy, cfg, smc_default):
    seed = cfg['detector']
    smc = smc_default if cfg['kgate'] == 4.0 else smc_k(noisy, cfg['kgate'])
    if cfg['baseline'] == 'interp':
        ib = de._interp_baseline(noisy, 3.0, 32)            # (nblk, nt)
        base = np.repeat(ib[:, None, :], GS, 1).reshape(-1, ib.shape[1])[:noisy.shape[0]]
    elif cfg['baseline'] == 'median':
        base = median_base(noisy)
    else:
        base = smc
    de2, _ = ind.iterate(noisy, n_iter=cfg['n_iter'], klo=cfg['klo'], khi=cfg['khi'],
                         dilate=cfg['dilate'], minc=cfg['minc'], seed=seed, base=base)
    return np.clip(de2, smc - cfg['clamp'], smc + cfg['clamp'])


def metrics(coh, noisy, s):
    cl = (noisy - coh).astype(np.float32); sig = np.abs(s) > 0
    ni = float(np.sqrt(np.mean((cl - s)[sig] ** 2))); no = float(np.sqrt(np.mean((cl - s)[~sig] ** 2)))
    r = cw.sparsify(cl, wavelet=CFG.wavelet, level=CFG.dwt_level, mode=CFG.dwt_mode, threshold=CFG.threshold_spec())
    rec = np.asarray(cw.reconstruct(r, noisy.shape[-1]))
    f0 = 1 - float(np.abs(rec - s)[sig].sum()) / max(float(np.abs(s)[sig].sum()), 1e-9)
    return f0, int(r.n_kept), ni, no


def main():
    knobs = sys.argv[1].split(',')
    n_ev = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    events = list(range(0, n_ev * 9, 9))
    planes = ['Y', 'U', 'V']
    # precompute per (plane,event): components + smc(kgate4)
    data = {p: [] for p in planes}
    for p in planes:
        for e in events:
            s, c, i = cc.components(p, e); noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            data[p].append((s, c, noisy, smc_k(noisy, 4.0)))
    for knob in knobs:
        print(f"\n########## KNOB: {knob} ##########", flush=True)
        for p in planes:
            print(f"  -- {p} --   {'val':>8} | {'F0':>7} {'kept':>7} {'nz_in':>6} {'nz_out':>6}")
            for v in SWEEP[knob]:
                cfg = dict(DEF); cfg[knob] = v
                rows = [metrics(build(no, cfg, smc), no, s) for (s, c, no, smc) in data[p]]
                a = np.array(rows).mean(0)
                vs = f"{v:g}" if not isinstance(v, str) else v
                print(f"  {'':>8}   {vs:>8} | {a[0]:>7.4f} {a[1]:>7.0f} {a[2]:>6.3f} {a[3]:>6.3f}", flush=True)


if __name__ == '__main__':
    main()
