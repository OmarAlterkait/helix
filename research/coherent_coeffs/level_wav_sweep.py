"""Does deeper decomposition / a different wavelet improve smart coherent removal?

Runs the level-aware magnitude-gate remover (smart.py) at many (wavelet, level) and
reports removal quality (F0 on signal, coh_left = leftover coherent RMS). Connects to
the earlier per-level finding (coarse bands signal-swamped; deeper just adds more such
bands) but now measured on the working estimator. coif3 caps at L7; short wavelets reach L9+.
"""
import os
import sys
import numpy as np
import pywt
import cc_common as cc
import smart as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
import common as wd  # noqa: E402

CONFIGS = [
    ('coif3', 4), ('coif3', 6), ('coif3', 7),
    ('sym8', 4), ('sym8', 6), ('sym8', 8),
    ('db4', 6), ('db4', 9),
    ('bior4.4', 4), ('bior4.4', 6), ('bior4.4', 8),
]


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    kg = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    events = list(range(0, n_ev * 13, 13))
    planes = ['Y', 'U', 'V']
    print(f"=== smart removal quality vs (wavelet, level), k={kg}, {n_ev} events ===")
    for p in planes:
        print(f"\n  plane {p}:")
        print(f"   {'wavelet/level':>16}  {'F0':>8}  {'coh_left':>9}")
        for wav, lev in CONFIGS:
            maxlev = pywt.dwt_max_level(cc.N_TICKS if hasattr(cc, 'N_TICKS') else 4321,
                                        pywt.Wavelet(wav).dec_len)
            L = min(lev, maxlev)
            rows = []
            for e in events:
                s, c, i = cc.components(p, e)
                noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
                cl, hc = sm.smart_removal(noisy, wavelet=wav, level=L, mode=cc.MODE, kgate=kg)
                rows.append(sm.metrics(cl, s, c, hc))
            f0, nr, cohl = np.array(rows).mean(0)
            tag = f"{wav} L{L}" + ("" if L == lev else f"(<{lev})")
            print(f"   {tag:>16}  {f0:>8.4f}  {cohl:>9.4f}")


if __name__ == '__main__':
    main()
