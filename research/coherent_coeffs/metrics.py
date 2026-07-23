"""Consolidated metrics for smart coherent removal vs raw/helix.
Removal stage (cleaned vs clean): F0, noise_rms, coh_left(leftover coherent RMS).
End-to-end (-> production sparsify): kept coeffs, compression, recon F0.
Per-plane recommended gate k (Y/V 3.5, U 3.0)."""
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
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402

KPLANE = {'Y': 3.5, 'U': 3.0, 'V': 3.5}


def removal_metrics(cleaned, signal, coherent, coh_hat):
    sig = np.abs(signal) > 0
    f0 = 1.0 - float(np.abs(cleaned - signal)[sig].sum()) / max(float(np.abs(signal)[sig].sum()), 1e-9)
    nrms = float(np.sqrt(np.mean((cleaned - signal)[~sig] ** 2)))
    cl = float(np.sqrt(np.mean((coh_hat - coherent) ** 2)))
    return f0, nrms, cl


def sp_metrics(cleaned, signal, cfg):
    res = cw.sparsify(cleaned, wavelet=cfg.wavelet, level=cfg.dwt_level,
                      mode=cfg.dwt_mode, threshold=cfg.threshold_spec())
    recon = np.asarray(cw.reconstruct(res, cleaned.shape[-1]))
    sig = np.abs(signal) > 0
    f0 = 1.0 - float(np.abs(recon - signal)[sig].sum()) / max(float(np.abs(signal)[sig].sum()), 1e-9)
    return int(res.n_kept), float(res.compression), f0


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    events = list(range(0, n_ev * 7, 7))
    cfg = DetectorConfig(group_size=64)
    for p in ['Y', 'U', 'V']:
        k = KPLANE[p]
        rows = {m: {'rem': [], 'sp': []} for m in ('raw', 'helix', 'smart')}
        for e in events:
            s, c, i = cc.components(p, e); ped = cc.PLANES[p]['pedestal']
            noisy = wd.digitize(s + c + i, ped)
            hel = np.asarray(remove_coherent(noisy, cfg))
            sm_img, sm_coh = sm.smart_removal(noisy, kgate=k)
            imgs = {'raw': (noisy, np.zeros_like(c)), 'helix': (hel, noisy - hel),
                    'smart': (sm_img, sm_coh)}
            for m, (img, coh_hat) in imgs.items():
                rows[m]['rem'].append(removal_metrics(img, s, c, coh_hat))
                rows[m]['sp'].append(sp_metrics(img, s, cfg))
        print(f"\n  ===== plane {p} (smart k={k}, {n_ev} events) =====")
        print(f"   {'method':>7} | {'F0_rem':>7} {'noise':>6} {'cohLeft':>7} | "
              f"{'kept':>7} {'comp':>7} {'F0_recon':>8}")
        for m in ('raw', 'helix', 'smart'):
            f0r, nr, cl = np.array(rows[m]['rem']).mean(0)
            kept, comp, f0s = np.array(rows[m]['sp']).mean(0)
            print(f"   {m:>7} | {f0r:>7.4f} {nr:>6.3f} {cl:>7.3f} | "
                  f"{kept:>7.0f} {comp:>6.1f}x {f0s:>8.4f}")


if __name__ == '__main__':
    main()
