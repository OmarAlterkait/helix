"""2x2 symlog panels for the smart coherent removal: clean / noisy(+coherent+intrinsic)
/ smart-removed / residual(removed-clean). Zoomed (signal-rich crop) and full plane.
Style matches the study: SymLogNorm(linthresh=2), RdBu_r, x=wire (64-wire block lines), y=tick."""
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
import common as wd  # noqa: E402
GS = cc.GROUP_SIZE


def _syml(ax, img, title, vmax, ws=None):
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='RdBu_r', norm=norm)
    # 64-wire block boundaries (x = wire)
    w0 = ws.start if ws else 0
    nwd = img.shape[0]
    for g in range(1, nwd // GS + 1):
        ax.axvline(g * GS, color='k', lw=0.3, alpha=0.35)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('wire' + (f' (+{w0})' if w0 else '')); ax.set_ylabel('tick')
    return im


def panel(plane, event, kgate=4.0, crop=True):
    s, c, i = cc.components(plane, event)
    nw, nt = s.shape
    noisy = wd.digitize(s + c + i, cc.PLANES[plane]['pedestal'])
    cleaned, _ = sm.smart_removal(noisy, kgate=kgate)
    if crop:
        e = np.abs(s).sum(1)
        wc = int(np.argmax(np.convolve(e, np.ones(5 * GS), 'same')))
        w0 = max(0, (wc // GS - 2) * GS); w1 = min(nw, w0 + 5 * GS); ws = slice(w0, w1)
        tc = int(np.argmax(np.abs(s[ws]).sum(0))); t0 = max(0, tc - 400); t1 = min(nt, t0 + 800); ts = slice(t0, t1)
        tag = 'zoom'
    else:
        ws = slice(0, nw); ts = slice(0, nt); tag = 'full'
    cl, no, rc, df = s[ws, ts], noisy[ws, ts], cleaned[ws, ts], (cleaned - s)[ws, ts]
    vmax = max(float(np.abs(cl).max()), 30.0)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    _syml(ax[0, 0], cl, 'clean (truth)', vmax, ws)
    _syml(ax[0, 1], no, 'noisy: +coherent +intrinsic', vmax, ws)
    _syml(ax[1, 0], rc, f'smart-removed (k={kgate:g})', vmax, ws)
    im = _syml(ax[1, 1], df, 'residual (removed - clean)', max(vmax * 0.4, 15), ws)
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(f'{plane} plane, event {event} — smart coherent removal, {tag} (symlog ADC)',
                 fontweight='bold')
    p = os.path.join(cc.FIGDIR, f'panel2x2_{plane}_e{event}_{tag}.png')
    fig.savefig(p, dpi=115, bbox_inches='tight'); plt.close(fig)
    print('saved', p)


def main():
    # (plane, event) pairs; a few events per plane, both zoom and full
    jobs = [('Y', 0), ('Y', 70), ('V', 0), ('V', 140)]
    for plane, e in jobs:
        panel(plane, e, crop=True)
        panel(plane, e, crop=False)


if __name__ == '__main__':
    main()
