"""Show de recovering coherent behind a dense track (Y), vs smart leaving/gating it."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc
import smart as sm
import detect_estimate as de

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
import common as wd  # noqa: E402
GS = cc.GROUP_SIZE


def main():
    plane = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    s, c, i = cc.components(plane, 0)
    nw, nt = s.shape
    noisy = wd.digitize(s + c + i, cc.PLANES[plane]['pedestal'])
    _, sm_coh = sm.smart_removal(noisy, kgate=4.0)
    _, de_coh = de.de_removal(noisy, baseline='smart')
    # densest block by max per-tick occupancy
    ng = cc.full_groups(nw)
    occ_max = [int((np.abs(s[g * GS:(g + 1) * GS]) > 5).sum(0).max()) for g in range(ng)]
    g0 = int(np.argmax(occ_max))
    lo = g0 * GS
    occ = (np.abs(s[lo:lo + GS]) > 5).sum(0)
    tc = int(np.argmax(occ)); t0 = max(0, tc - 300); t1 = min(nt, t0 + 600); tw = slice(t0, t1)
    tt = np.arange(t0, t1)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(tt, c[lo, tw], 'k', lw=2.4, label='true coherent', zorder=5)
    ax[0].plot(tt, sm_coh[lo, tw], 'tab:blue', lw=1.3, label='smart estimate')
    ax[0].plot(tt, de_coh[lo, tw], 'tab:red', lw=1.3, label='de estimate')
    ax[0].set_title(f'{plane} block {g0} (max occupancy {occ.max()}/64): coherent estimate '
                    f'through a dense track')
    ax[0].set_ylabel('coherent ADC'); ax[0].legend(ncol=3, fontsize=9)
    ax2 = ax[0].twinx(); ax2.fill_between(tt, occ[tw], color='gray', alpha=0.15)
    ax2.set_ylabel('occupancy', color='gray')
    # estimate error
    ax[1].plot(tt, sm_coh[lo, tw] - c[lo, tw], 'tab:blue', lw=1.0,
               label=f'smart err (RMS {np.sqrt(np.mean((sm_coh[lo,tw]-c[lo,tw])**2)):.2f})')
    ax[1].plot(tt, de_coh[lo, tw] - c[lo, tw], 'tab:red', lw=1.0,
               label=f'de err (RMS {np.sqrt(np.mean((de_coh[lo,tw]-c[lo,tw])**2)):.2f})')
    ax[1].axhline(0, color='k', lw=0.5); ax[1].set_ylabel('estimate - true (ADC)')
    ax[1].set_xlabel('tick'); ax[1].legend(fontsize=9)
    fig.suptitle(f'{plane}: de recovers coherent behind the dense track where smart gates off',
                 fontweight='bold')
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, f'fig15_de_recover_{plane}.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print('saved', p)


if __name__ == '__main__':
    main()
