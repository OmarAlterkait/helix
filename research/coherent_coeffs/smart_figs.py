"""Visuals for the smart level-aware gated coherent removal.

fig9 : grouped-bar comparison (raw / helix / smart) of F0, noise_rms, coh_left per plane.
fig10: per plane, top row sample-space symlog (clean | noisy+coherent | smart-removed |
       residual), bottom row A4 coeff map (noisy stripes | gated coherent estimate |
       after removal) — shows block stripes removed while signal streaks survive.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

import cc_common as cc
import smart as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('numpy')
from helix.tpc.config import DetectorConfig  # noqa: E402
from helix.tpc.coherent import remove_coherent  # noqa: E402
GS = cc.GROUP_SIZE


def fig_compare(n_ev=6, kgate=4.0):
    planes = ['Y', 'U', 'V']
    cfg = DetectorConfig(group_size=64)
    events = list(range(0, n_ev * 37, 37))
    res = {p: {'raw': [], 'helix': [], 'smart': []} for p in planes}
    for p in planes:
        for e in events:
            s, c, i = cc.components(p, e)
            noisy = wd.digitize(s + c + i, cc.PLANES[p]['pedestal'])
            res[p]['raw'].append(sm.metrics(noisy, s, c, np.zeros_like(c)))
            hel = np.asarray(remove_coherent(noisy, cfg))
            res[p]['helix'].append(sm.metrics(hel, s, c, noisy - hel))
            cl, hc = sm.smart_removal(noisy, kgate=kgate)
            res[p]['smart'].append(sm.metrics(cl, s, c, hc))
    metr = ['F0 (higher better)', 'noise_rms (lower)', 'coh_left (lower)']
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(planes)); w = 0.26
    cols = {'raw': 'lightgray', 'helix': 'tab:gray', 'smart': 'tab:red'}
    for mi in range(3):
        for j, m in enumerate(['raw', 'helix', 'smart']):
            vals = [np.array(res[p][m]).mean(0)[mi] for p in planes]
            ax[mi].bar(x + (j - 1) * w, vals, w, label=m if mi == 0 else None, color=cols[m])
        ax[mi].set_xticks(x); ax[mi].set_xticklabels(planes)
        ax[mi].set_title(metr[mi])
    ax[0].set_ylim(0.84, 0.98); ax[0].legend()
    fig.suptitle(f'Smart level-aware gated removal (k={kgate}) vs helix — {n_ev} events',
                 fontweight='bold')
    fig.tight_layout()
    pth = os.path.join(cc.FIGDIR, 'fig9_smart_vs_helix.png')
    fig.savefig(pth, dpi=130); plt.close(fig)
    print('saved', pth)


def _syml(ax, img, title, vmax, blocks=False, nblk=0):
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='RdBu_r', norm=norm)
    if blocks:
        for g in range(1, nblk):
            ax.axvline(g * GS, color='k', lw=0.3, alpha=0.4)
    ax.set_title(title, fontsize=9)
    return im


def fig_panel(ptype='V', event=0, kgate=4.0):
    s, c, i = cc.components(ptype, event)
    nw, nt = s.shape
    noisy = wd.digitize(s + c + i, cc.PLANES[ptype]['pedestal'])
    cleaned, coh_hat = sm.smart_removal(noisy, kgate=kgate)
    # signal-rich crop
    e = np.abs(s).sum(1)
    wc = int(np.argmax(np.convolve(e, np.ones(5 * GS), 'same')))
    w0 = max(0, wc - 2 * GS); w1 = min(nw, w0 + 5 * GS); ws = slice(w0, w1)
    tc = int(np.argmax(np.abs(s[ws]).sum(0))); t0 = max(0, tc - 350); t1 = min(nt, t0 + 700); ts = slice(t0, t1)
    nblk_crop = (w1 - w0) // GS + 1

    fig, ax = plt.subplots(2, 4, figsize=(17, 8))
    vmax = max(float(np.abs(s[ws, ts]).max()), 30.0)
    _syml(ax[0, 0], s[ws, ts], 'clean (truth)', vmax, True, nblk_crop)
    _syml(ax[0, 1], noisy[ws, ts], 'noisy: +coherent +intrinsic', vmax, True, nblk_crop)
    _syml(ax[0, 2], cleaned[ws, ts], f'smart-removed (k={kgate})', vmax, True, nblk_crop)
    im = _syml(ax[0, 3], (cleaned - s)[ws, ts], 'residual (removed - clean)', max(vmax * 0.4, 15))
    for a in ax[0]:
        a.set_xlabel('wire'); a.set_ylabel('tick')
    fig.colorbar(im, ax=ax[0, 3], fraction=0.046)

    # bottom: A4 coeff map noisy / gated coherent estimate / after
    bn = cc.dwt_bands(noisy)[0][ws]
    be = cc.dwt_bands(coh_hat)[0][ws]    # coh_hat decomposed -> its A4 = gated estimate
    ba = bn - be
    vc = max(float(np.percentile(np.abs(bn), 99.5)), 5.0)
    nrm = SymLogNorm(linthresh=max(vc * 0.05, 1.0), vmin=-vc, vmax=vc, base=10)
    for k, (band, title) in enumerate([(bn, 'A4 coeffs: noisy (block stripes)'),
                                        (be, 'A4: gated coherent estimate'),
                                        (ba, 'A4: after removal (stripes gone)')]):
        im2 = ax[1, k].imshow(band.T, aspect='auto', origin='lower', cmap='RdBu_r', norm=nrm)
        for g in range(1, nblk_crop):
            ax[1, k].axvline(g * GS, color='k', lw=0.3, alpha=0.4)
        ax[1, k].set_title(title, fontsize=9); ax[1, k].set_xlabel('wire'); ax[1, k].set_ylabel('A4 pos')
    fig.colorbar(im2, ax=ax[1, 2], fraction=0.046)
    ax[1, 3].axis('off')
    ax[1, 3].text(0.05, 0.5, 'coherent block-stripes\nremoved in A4;\nsignal streaks preserved\n\n'
                  'gate keeps |m|<k*sigma_coh\n(small dense = coherent),\ndrops large (sparse = signal)',
                  fontsize=11, va='center')
    fig.suptitle(f'{ptype} plane — smart level-aware gated coherent removal (symlog ADC)',
                 fontweight='bold')
    fig.tight_layout()
    pth = os.path.join(cc.FIGDIR, f'fig10_smart_panel_{ptype}.png')
    fig.savefig(pth, dpi=120); plt.close(fig)
    print('saved', pth)


if __name__ == '__main__':
    fig_compare()
    for p in ('V', 'Y'):
        fig_panel(p)
