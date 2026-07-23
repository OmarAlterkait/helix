"""Full-plane coefficient maps at the 4 production levels (coif3 L4), both cases:
  case A = signal + intrinsic            (NO coherent)
  case B = signal + intrinsic + coherent (WITH coherent)

Each level (band) on its OWN symlog scale, because the per-level amplitudes differ
by ~10x (A4 sigma ~7.6 vs D1 ~0.9) — the reason a single mask threshold fails.
x = wire (black lines every 64-wire block), y = coeff position within the band.
Coherent shows up in case B as block-constant vertical stripes; signal as sparse
streaks; intrinsic as speckle. This is the substrate for a smarter cross-wire /
cross-block estimator.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

import cc_common as cc
GS = cc.GROUP_SIZE


def main():
    ptype = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    event = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    signal, coherent, intrinsic = cc.components(ptype, event)
    nw = signal.shape[0]
    ped = cc.PLANES[ptype]['pedestal']

    caseA = signal + intrinsic                 # no coherent
    caseB = signal + intrinsic + coherent      # with coherent
    bA = cc.dwt_bands(caseA)
    bB = cc.dwt_bands(caseB)
    labels = cc.band_labels()
    nblk = cc.n_groups(nw)

    fig, ax = plt.subplots(len(labels), 2, figsize=(15, 3.0 * len(labels)))
    for r, lab in enumerate(labels):
        # shared per-band scale from the with-coherent case (robust)
        vmax = max(float(np.percentile(np.abs(bB[r]), 99.5)), 3.0)
        norm = SymLogNorm(linthresh=max(vmax * 0.05, 1.0), vmin=-vmax, vmax=vmax, base=10)
        for c, (band, cname) in enumerate([(bA[r], 'no coherent'), (bB[r], 'WITH coherent')]):
            im = ax[r, c].imshow(band.T, aspect='auto', origin='lower', cmap='RdBu_r', norm=norm)
            for g in range(1, nblk):
                ax[r, c].axvline(g * GS, color='k', lw=0.3, alpha=0.4)
            ax[r, c].set_ylabel(f'{lab}\ncoeff pos', fontsize=9)
            ax[r, c].set_title(f'{lab}: {cname}  (|coeff| scale +/-{vmax:.1f})', fontsize=9)
            if r == len(labels) - 1:
                ax[r, c].set_xlabel('wire')
            fig.colorbar(im, ax=ax[r, c], fraction=0.04, pad=0.02)
    fig.suptitle(f'{ptype} plane — full-plane coefficient maps, {cc.WAVELET} L{cc.LEVEL} '
                 f'(left: no coherent | right: + coherent)\n'
                 f'coherent = block-constant vertical stripes; note the ~10x amplitude '
                 f'drop coarse(A4)->fine(D1)', fontweight='bold', fontsize=11)
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, f'fig8_planes_levels_{ptype}.png')
    fig.savefig(p, dpi=110); plt.close(fig)
    print('saved', p)


if __name__ == '__main__':
    main()
