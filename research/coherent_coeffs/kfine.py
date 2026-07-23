"""Fine k-sweep: smart-removal -> sparsify (F0, kept) frontier vs helix & oracle.
Maps the F0-vs-coefficient tradeoff so we can pick the best per-plane gate k."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import sys
import numpy as np
import cc_common as cc
import smart as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('jax')
from helix.core import wavelet as cw  # noqa: E402
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402


def spcount(cleaned, signal, cfg):
    res = cw.sparsify(cleaned, wavelet=cfg.wavelet, level=cfg.dwt_level,
                      mode=cfg.dwt_mode, threshold=cfg.threshold_spec())
    recon = np.asarray(cw.reconstruct(res, cleaned.shape[-1]))
    sig = np.abs(signal) > 0
    f0 = 1.0 - float(np.abs(recon - signal)[sig].sum()) / max(float(np.abs(signal)[sig].sum()), 1e-9)
    return res.n_kept, f0


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    events = list(range(0, n_ev * 9, 9))
    cfg = DetectorConfig(group_size=64)
    ks = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    for p in ['Y', 'U', 'V']:
        hel, ks_kept, ks_f0 = [], {k: [] for k in ks}, {k: [] for k in ks}
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            hel.append(spcount(np.asarray(remove_coherent(noisy, cfg)), s, cfg))
            for k in ks:
                kept, f0 = spcount(sm.smart_removal(noisy, kgate=k)[0], s, cfg)
                ks_kept[k].append(kept); ks_f0[k].append(f0)
        hk, hf = np.array(hel).mean(0)
        print(f"\n  plane {p}: helix kept {hk:.0f} F0 {hf:.4f}")
        print(f"   {'k':>4}  {'kept':>8}  {'F0':>8}  {'vs helix kept':>13}")
        for k in ks:
            kk = np.mean(ks_kept[k]); ff = np.mean(ks_f0[k])
            print(f"   {k:>4}  {kk:>8.0f}  {ff:>8.4f}  {100*(kk-hk)/hk:>+12.1f}%")


if __name__ == '__main__':
    main()
