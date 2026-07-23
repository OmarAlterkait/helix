"""End-to-end coefficient count: does smart coherent removal compress better?

Each path produces a cleaned image, then the SAME production sparsify
(per-band sigma + threshold-approx, coif3 L4, kappa=1) thresholds it. We report
kept-coeff count, compression (pixels/kept), and reconstruction F0 vs clean signal.
Fewer kept coeffs at equal/higher F0 = better. Coherent that survives removal
inflates coarse-band sigma and is kept as wasted coefficients.

  raw    : no removal (baseline; coherent leaks into kept coeffs)
  helix  : sample-space multi-pass removal
  smart  : level-aware gated common-mode removal (smart.py)
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import sys
import numpy as np

import cc_common as cc
import smart as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('jax')   # GPU: helix removal + sparsify (smart_removal uses pywt independently)
from helix.core import wavelet as cw  # noqa: E402
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402


def sparsify_count(cleaned, signal, cfg):
    """Production sparsify -> (kept, total, compression, F0 vs clean signal)."""
    res = cw.sparsify(cleaned, wavelet=cfg.wavelet, level=cfg.dwt_level,
                      mode=cfg.dwt_mode, threshold=cfg.threshold_spec())
    recon = np.asarray(cw.reconstruct(res, cleaned.shape[-1]))
    sig = np.abs(signal) > 0
    tc = float(np.abs(signal)[sig].sum())
    f0 = 1.0 - float(np.abs(recon - signal)[sig].sum()) / max(tc, 1e-9)
    return res.n_kept, res.n_total, res.compression, f0


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    kgate = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    planes = ['Y', 'U', 'V']
    events = list(range(0, n_ev * 7, 7))
    cfg = DetectorConfig(group_size=64)
    kgates = [3.0, 3.5, 4.0]
    methods = ['raw', 'helix'] + [f'smart_k{k}' for k in kgates]

    print(f"=== coefficient count, {n_ev} events, "
          f"{cfg.wavelet} L{cfg.dwt_level} per-band-sigma+approx ===")
    for p in planes:
        agg = {m: {'kept': [], 'comp': [], 'f0': []} for m in methods}
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            imgs = {'raw': noisy, 'helix': np.asarray(remove_coherent(noisy, cfg))}
            for k in kgates:
                imgs[f'smart_k{k}'] = sm.smart_removal(noisy, kgate=k)[0]
            for m in methods:
                k, tot, comp, f0 = sparsify_count(imgs[m], s, cfg)
                agg[m]['kept'].append(k); agg[m]['comp'].append(comp); agg[m]['f0'].append(f0)
        print(f"\n  --- plane {p} (n_total/plane = {tot}) ---")
        print(f"   {'method':>8}  {'kept/plane':>11}  {'compression':>11}  {'F0':>8}")
        for m in methods:
            kept = np.mean(agg[m]['kept']); comp = np.mean(agg[m]['comp']); f0 = np.mean(agg[m]['f0'])
            print(f"   {m:>8}  {kept:>11.0f}  {comp:>10.1f}x  {f0:>8.4f}")
        # smart vs helix deltas
        for k in kgates:
            m = f'smart_k{k}'
            dk = 100 * (np.mean(agg[m]['kept']) - np.mean(agg['helix']['kept'])) / np.mean(agg['helix']['kept'])
            df = np.mean(agg[m]['f0']) - np.mean(agg['helix']['f0'])
            print(f"   {m} vs helix: kept {dk:+.1f}%, F0 {df:+.4f}")


if __name__ == '__main__':
    main()
