"""de (detect-then-estimate, smart baseline) vs helix/smart/oracle: removal + count."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import sys
import numpy as np
import cc_common as cc
import smart as sm
import detect_estimate as de

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('jax')
from helix.core import wavelet as cw  # noqa: E402
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402


def rem_m(cleaned, s, c, coh_hat):
    sig = np.abs(s) > 0
    f0 = 1 - float(np.abs(cleaned - s)[sig].sum()) / max(float(np.abs(s)[sig].sum()), 1e-9)
    return f0, float(np.sqrt(np.mean((coh_hat - c) ** 2)))


def sp_m(cleaned, s, cfg):
    res = cw.sparsify(cleaned, wavelet=cfg.wavelet, level=cfg.dwt_level, mode=cfg.dwt_mode,
                      threshold=cfg.threshold_spec())
    recon = np.asarray(cw.reconstruct(res, cleaned.shape[-1]))
    sig = np.abs(s) > 0
    f0 = 1 - float(np.abs(recon - s)[sig].sum()) / max(float(np.abs(s)[sig].sum()), 1e-9)
    return int(res.n_kept), f0


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    events = list(range(0, n_ev * 7, 7))
    cfg = DetectorConfig(group_size=64)
    methods = ['helix', 'smart', 'de_k1.5', 'de_k2.0', 'oracle']
    for p in ['Y', 'U', 'V']:
        agg = {m: {'rem': [], 'sp': []} for m in methods}
        for e in events:
            s, c, i = cc.components(p, e); ped = cc.PLANES[p]['pedestal']
            noisy = wd.digitize(s + c + i, ped)
            hel = np.asarray(remove_coherent(noisy, cfg))
            sm_img, sm_coh = sm.smart_removal(noisy, kgate=4.0)
            d15_img, d15_coh = de.de_removal(noisy, baseline='smart', ksig=1.5)
            d20_img, d20_coh = de.de_removal(noisy, baseline='smart', ksig=2.0)
            orc = wd.digitize(s + i, ped)
            imgs = {'helix': (hel, noisy - hel), 'smart': (sm_img, sm_coh),
                    'de_k1.5': (d15_img, d15_coh), 'de_k2.0': (d20_img, d20_coh), 'oracle': (orc, c)}
            for m, (img, ch) in imgs.items():
                agg[m]['rem'].append(rem_m(img, s, c, ch))
                agg[m]['sp'].append(sp_m(img, s, cfg))
        print(f"\n  ===== plane {p} ({n_ev} events) =====")
        print(f"   {'method':>9} | {'F0_rem':>7} {'cohLeft':>7} | {'kept':>7} {'F0_recon':>8}")
        for m in methods:
            f0r, cl = np.array(agg[m]['rem']).mean(0)
            kept, f0s = np.array(agg[m]['sp']).mean(0)
            print(f"   {m:>9} | {f0r:>7.4f} {cl:>7.3f} | {kept:>7.0f} {f0s:>8.4f}")


if __name__ == '__main__':
    main()
