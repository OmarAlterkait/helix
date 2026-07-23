"""Validate the best induction method (iterate hysteresis-amp, dilate 11) over N events,
all planes, vs smart and the true-mask oracle. Reports F0_recon, coh_left, kept."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import sys
import numpy as np
import cc_common as cc
import smart as sm
import induction as ind

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
import common as wd  # noqa: E402


def main():
    n_ev = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    events = list(range(0, n_ev * 9, 9))
    for p in ['Y', 'U', 'V']:
        agg = {m: [] for m in ['smart', 'de2', 'oracle']}
        for e in events:
            s, c, i = cc.components(p, e); noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            _, smc = sm.smart_removal(noisy, kgate=4.0)
            agg['smart'].append(ind.full_metrics(noisy, smc, s, c))
            coh, _ = ind.iterate(noisy, n_iter=4, klo=0.7, khi=3.5, dilate=15, seed='amp')
            agg['de2'].append(ind.full_metrics(noisy, coh, s, c))
            agg['oracle'].append(ind.full_metrics(noisy, ind.estimate(noisy, ind.mask_true(s)), s, c))
        print(f"\n  ===== {p} ({n_ev} ev) =====   F0_rem  cohLeft   kept  F0_recon")
        for m in ['smart', 'de2', 'oracle']:
            a = np.array(agg[m]).mean(0)
            print(f"   {m:>7}   {a[0]:.4f}  {a[1]:.3f}  {a[2]:.0f}  {a[3]:.4f}")


if __name__ == '__main__':
    main()
