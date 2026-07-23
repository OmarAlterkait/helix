"""2x2 symlog panels of the FULL pipeline: clean / noisy / (de removal -> sparsify ->
reconstruct) / residual. de = detect-then-estimate coherent removal (ksig 1.5), then the
production sparsify (per-band sigma + threshold-approx, coif3 L4). Zoom + full, a few events.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

import cc_common as cc
import detect_estimate as de

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import common as wd  # noqa: E402
from helix.core import backend as _be  # noqa: E402
_be.set_backend('numpy')
from helix.core import wavelet as cw  # noqa: E402
from helix.tpc.config import DetectorConfig  # noqa: E402
GS = cc.GROUP_SIZE
CFG = DetectorConfig(group_size=64)


def _syml(ax, img, title, vmax, nblk_lines=0):
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(img.T, aspect='auto', origin='lower', cmap='RdBu_r', norm=norm)
    for g in range(1, nblk_lines):
        ax.axvline(g * GS, color='k', lw=0.3, alpha=0.35)
    ax.set_title(title, fontsize=10); ax.set_xlabel('wire'); ax.set_ylabel('tick')
    return im


def panel(plane, event, ksig=1.5, crop=True):
    s, c, i = cc.components(plane, event)
    nw, nt = s.shape
    noisy = wd.digitize(s + c + i, cc.PLANES[plane]['pedestal'])
    cleaned, _ = de.de_removal(noisy, baseline='smart', ksig=ksig)
    res = cw.sparsify(cleaned, wavelet=CFG.wavelet, level=CFG.dwt_level,
                      mode=CFG.dwt_mode, threshold=CFG.threshold_spec())
    recon = np.asarray(cw.reconstruct(res, nt)).astype(np.float32)
    sig = np.abs(s) > 0
    f0 = 1.0 - float(np.abs(recon - s)[sig].sum()) / max(float(np.abs(s)[sig].sum()), 1e-9)
    comp = res.compression
    if crop:
        e = np.abs(s).sum(1); wc = int(np.argmax(np.convolve(e, np.ones(5 * GS), 'same')))
        w0 = max(0, (wc // GS - 2) * GS); w1 = min(nw, w0 + 5 * GS); ws = slice(w0, w1)
        tc = int(np.argmax(np.abs(s[ws]).sum(0))); t0 = max(0, tc - 400); t1 = min(nt, t0 + 800); ts = slice(t0, t1)
        tag = 'zoom'; nbl = (w1 - w0) // GS + 1
    else:
        ws = slice(0, nw); ts = slice(0, nt); tag = 'full'; nbl = 0
    cl, no, rc, df = s[ws, ts], noisy[ws, ts], recon[ws, ts], (recon - s)[ws, ts]
    vmax = max(float(np.abs(cl).max()), 30.0)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    _syml(ax[0, 0], cl, 'clean (truth)', vmax, nbl)
    _syml(ax[0, 1], no, 'noisy: +coherent +intrinsic', vmax, nbl)
    _syml(ax[1, 0], rc, f'de removal -> sparsify -> reconstruct  (~{comp:.0f}x)', vmax, nbl)
    im = _syml(ax[1, 1], df, 'residual (reconstruction - clean)', max(vmax * 0.4, 15), 0)
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.suptitle(f'{plane} plane, event {event} — FULL pipeline (de k{ksig} + sparsify), {tag}  '
                 f'[F0 {f0:.3f}, ~{comp:.0f}x]  symlog ADC', fontweight='bold')
    p = os.path.join(cc.FIGDIR, f'panelfull_{plane}_e{event}_{tag}.png')
    fig.savefig(p, dpi=115, bbox_inches='tight'); plt.close(fig)
    print(f'saved {p}  F0 {f0:.4f} comp {comp:.0f}x')


def main():
    jobs = [('Y', 0), ('Y', 70), ('V', 0), ('U', 0)]
    for plane, e in jobs:
        panel(plane, e, crop=True)
        panel(plane, e, crop=False)


if __name__ == '__main__':
    main()
