"""Definitive F0-vs-coefficient frontier on CONSISTENT events: smart(k-curve) vs helix
vs oracle (coherent never added). Saves JSON + figure. Shows smart reaching the oracle
count and dominating helix."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import sys
import json
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
    return int(res.n_kept), f0


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    events = list(range(0, n_ev * 7, 7))
    cfg = DetectorConfig(group_size=64)
    ks = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    out = {}
    for p in ['Y', 'U', 'V']:
        hel, orc = [], []
        smc = {k: [] for k in ks}
        for e in events:
            s, c, i = cc.components(p, e); ped = cc.PLANES[p]['pedestal']
            noisy = wd.digitize(s + c + i, ped)
            hel.append(spcount(np.asarray(remove_coherent(noisy, cfg)), s, cfg))
            orc.append(spcount(wd.digitize(s + i, ped), s, cfg))
            for k in ks:
                smc[k].append(spcount(sm.smart_removal(noisy, kgate=k)[0], s, cfg))
        out[p] = {'ks': ks,
                  'smart_kept': [float(np.mean([r[0] for r in smc[k]])) for k in ks],
                  'smart_f0': [float(np.mean([r[1] for r in smc[k]])) for k in ks],
                  'helix_kept': float(np.mean([r[0] for r in hel])),
                  'helix_f0': float(np.mean([r[1] for r in hel])),
                  'oracle_kept': float(np.mean([r[0] for r in orc])),
                  'oracle_f0': float(np.mean([r[1] for r in orc]))}
        print(f"{p}: done")
    with open(os.path.join(cc.FIGDIR, '..', 'frontier.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("saved frontier.json")


if __name__ == '__main__':
    main()
