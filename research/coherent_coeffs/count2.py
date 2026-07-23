"""Best end-to-end: smart removal (coif3 L4) then sparsify with the compression-optimal
wavelet. Decouples removal-transform from sparsify-transform. The cleaned IMAGE can be
sparsified by any wavelet; the prior study found bior4.4 L8 best for sparsification.
Compares sparsify @ coif3-L4 vs bior4.4-L8 on helix- and smart-cleaned images.
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
_be.set_backend('jax')
from helix.core import wavelet as cw  # noqa: E402
from dataclasses import replace  # noqa: E402
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402


def spcount(cleaned, signal, scfg):
    res = cw.sparsify(cleaned, wavelet=scfg.wavelet, level=scfg.dwt_level,
                      mode=scfg.dwt_mode, threshold=scfg.threshold_spec())
    recon = np.asarray(cw.reconstruct(res, cleaned.shape[-1]))
    sig = np.abs(signal) > 0
    f0 = 1.0 - float(np.abs(recon - signal)[sig].sum()) / max(float(np.abs(signal)[sig].sum()), 1e-9)
    return res.n_kept, res.compression, f0


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    events = list(range(0, n_ev * 11, 11))
    rem_cfg = DetectorConfig(group_size=64)                          # helix removal cfg
    sp_cfgs = {'coif3L4': rem_cfg,
               'bior4.4L8': replace(rem_cfg, wavelet='bior4.4', dwt_level=8)}
    for p in ['Y', 'U', 'V']:
        agg = {(rm, sp): [] for rm in ('helix', 'smart') for sp in sp_cfgs}
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            cleaned = {'helix': np.asarray(remove_coherent(noisy, rem_cfg)),
                       'smart': sm.smart_removal(noisy, kgate=4.0)[0]}
            for rm in cleaned:
                for sp, scfg in sp_cfgs.items():
                    agg[(rm, sp)].append(spcount(cleaned[rm], s, scfg))
        print(f"\n  plane {p}:")
        print(f"   {'removal/sparsify':>22}  {'kept':>8}  {'comp':>8}  {'F0':>8}")
        for (rm, sp), v in agg.items():
            k, comp, f0 = np.array(v).mean(0)
            print(f"   {rm+' / '+sp:>22}  {k:>8.0f}  {comp:>7.1f}x  {f0:>8.4f}")


if __name__ == '__main__':
    main()
