"""Figures for the coherent-coefficient structure study.

Decomposes signal / coherent / intrinsic SEPARATELY (no thresholding) and
visualizes the spatial + multi-level structure that motivates a coefficient-space
coherent estimator. Symlog ADC style (SymLogNorm linthresh=2, RdBu_r) matching
the rest of the study.

Figures:
  1 coherent_within_vs_across   coherent waveform: identical within a block,
                                 anti-correlated across adjacent blocks
  2 signal_adjacent_wires        adjacent signal wires vary (vs coherent identity)
  3 decomp_levels                per-level coeff bands, signal vs coherent vs intrinsic
  4 coeff_maps                   wire x coeff heatmaps per band: coherent = blocky
                                 stripes, intrinsic = speckle, signal = sparse streaks
  5 level_energy                 energy% + MAD-sigma per level, 3 components
  6 cross_block_corr             block-to-block coherent correlation (bleeding)
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


def _busiest_block(signal):
    """Index of the full 64-block carrying the most signal energy."""
    nw = signal.shape[0]
    ng = cc.full_groups(nw)
    e = [float(np.sum(signal[g * GS:(g + 1) * GS] ** 2)) for g in range(ng)]
    return int(np.argmax(e))


# ---------------------------------------------------------------- Fig 1
def fig_within_vs_across(coherent, ptype, g0=2, tw=(1000, 2600)):
    t = np.arange(*tw)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    # within one block: 4 wires of block g0 — perfectly coincide
    base = g0 * GS
    offs = [0, 21, 42, 63]
    styles = [('k', 2.6, 1.0, '-'), ('tab:red', 1.4, 0.9, '--'),
              ('tab:green', 1.4, 0.9, ':'), ('tab:blue', 1.4, 0.9, '-.')]
    for o, (c, lw, a, ls) in zip(offs, styles):
        ax[0].plot(t, coherent[base + o, slice(*tw)], color=c, lw=lw, alpha=a, ls=ls,
                   label=f'wire {base + o}')
    blk = coherent[base:base + GS, slice(*tw)]
    maxdiff = float(np.max(np.abs(blk - blk[0:1])))
    ax[0].set_title(f'WITHIN one 64-wire block (block {g0}): all wires share the '
                    f'identical coherent waveform  —  max pairwise diff = {maxdiff:.1e} ADC',
                    fontsize=10.5)
    ax[0].legend(ncol=4, fontsize=8, loc='upper right')
    ax[0].set_ylabel('coherent ADC')
    # across blocks: one wire from each of 4 adjacent blocks
    cols = ['k', 'tab:red', 'tab:green', 'tab:blue']
    for k, c in enumerate(cols):
        g = g0 + k
        ax[1].plot(t, coherent[g * GS, slice(*tw)], color=c, lw=1.4, alpha=0.85,
                   label=f'block {g} (wire {g * GS})')
    # annotate neighbor anti-correlation
    a0 = coherent[g0 * GS, slice(*tw)]; a1 = coherent[(g0 + 1) * GS, slice(*tw)]
    cc01 = float(np.corrcoef(a0, a1)[0, 1])
    ax[1].set_title(f'ACROSS adjacent blocks: each block differs; neighbors '
                    f'anti-correlated (corr block{g0}/block{g0+1} = {cc01:+.2f}, beta coupling)',
                    fontsize=10.5)
    ax[1].legend(ncol=4, fontsize=8, loc='upper right')
    ax[1].set_xlabel('tick'); ax[1].set_ylabel('coherent ADC')
    fig.suptitle(f'{ptype} plane — coherent noise structure (no thresholding)', fontweight='bold')
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, f'fig1_within_vs_across_{ptype}.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print('saved', p, f'(within-block maxdiff {maxdiff:.1e}, neighbor corr {cc01:+.2f})')


# ---------------------------------------------------------------- Fig 2
def fig_signal_adjacent(signal, coherent, ptype):
    g0 = _busiest_block(signal)
    base = g0 * GS
    # find the tick window with the pulse
    blk = signal[base:base + GS]
    tcen = int(np.argmax(np.abs(blk).sum(0)))
    tw = (max(0, tcen - 300), min(signal.shape[1], tcen + 300))
    t = np.arange(*tw)
    # pick 5 adjacent wires that actually carry signal in this window
    en = np.abs(signal[base:base + GS, slice(*tw)]).sum(1)
    w_start = base + max(0, int(np.argmax(en)) - 2)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for k in range(5):
        w = w_start + k
        ax[0].plot(t, signal[w, slice(*tw)], lw=1.3, alpha=0.85, label=f'wire {w}')
    ax[0].set_title(f'SIGNAL on 5 adjacent wires (block {g0}): each wire differs '
                    f'(track crosses wires at different times)', fontsize=10.5)
    ax[0].legend(ncol=5, fontsize=8); ax[0].set_ylabel('signal ADC')
    for k in range(5):
        w = w_start + k
        ax[1].plot(t, coherent[w, slice(*tw)], lw=1.3, alpha=0.85, label=f'wire {w}')
    ax[1].set_title('COHERENT on the same 5 wires: identical (all in one block)', fontsize=10.5)
    ax[1].legend(ncol=5, fontsize=8); ax[1].set_xlabel('tick'); ax[1].set_ylabel('coherent ADC')
    fig.suptitle(f'{ptype} plane — signal varies wire-to-wire, coherent does not',
                 fontweight='bold')
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, f'fig2_signal_adjacent_{ptype}.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print('saved', p)


# ---------------------------------------------------------------- Fig 3
def fig_decomp_levels(signal, coherent, intrinsic, ptype):
    g0 = _busiest_block(signal)
    w = g0 * GS + 5
    comps = [('signal', signal), ('coherent', coherent), ('intrinsic', intrinsic)]
    labels = cc.band_labels()
    nlev = len(labels)
    fig, ax = plt.subplots(nlev, 3, figsize=(13, 9))
    for c, (cname, img) in enumerate(comps):
        bands = cc.dwt_bands(img[w:w + 1])           # single wire
        for r, (b, lab) in enumerate(zip(bands, labels)):
            v = b[0]
            ax[r, c].plot(v, lw=0.7, color='tab:blue')
            ax[r, c].axhline(0, color='k', lw=0.4, alpha=0.4)
            ax[r, c].set_ylabel(lab, fontsize=9)
            if r == 0:
                ax[r, c].set_title(f'{cname}  (wire {w})', fontsize=11, fontweight='bold')
            ax[r, c].tick_params(labelsize=7)
            # share y per row across components for visual comparison
        # set per-row shared ylim
    for r in range(nlev):
        ymax = max(np.abs(ax[r, c].get_ylim()).max() for c in range(3))
        for c in range(3):
            ax[r, c].set_ylim(-ymax, ymax)
    for c in range(3):
        ax[-1, c].set_xlabel('coeff position', fontsize=9)
    fig.suptitle(f'{ptype} plane — per-level DWT bands ({cc.WAVELET} L{cc.LEVEL}), one wire, '
                 f'no thresholding\n(coarse A4/D4 carry signal AND coherent; intrinsic spreads to fine bands)',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, f'fig3_decomp_levels_{ptype}.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print('saved', p)


# ---------------------------------------------------------------- Fig 4
def _coeff_map(ax, band, title, nblocks, vmax):
    """band is (n_wires_crop, len_band); show as (coeff x wire) symlog image."""
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(band.T, aspect='auto', origin='lower', cmap='RdBu_r', norm=norm)
    for g in range(1, nblocks):
        ax.axvline(g * GS, color='k', lw=0.5, alpha=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('wire'); ax.set_ylabel('coeff position')
    return im


def fig_coeff_maps(signal, coherent, intrinsic, ptype, bands_to_show=('A4', 'D2')):
    g0 = _busiest_block(signal)
    nblocks = 5
    w0 = max(0, (g0 - 1)) * GS
    w1 = w0 + nblocks * GS
    labels = cc.band_labels()
    comps = [('signal', signal), ('coherent', coherent), ('intrinsic', intrinsic)]
    bidx = [labels.index(b) for b in bands_to_show]
    fig, ax = plt.subplots(len(bands_to_show), 3, figsize=(13, 4.2 * len(bands_to_show)))
    if len(bands_to_show) == 1:
        ax = ax[None, :]
    for ci, (cname, img) in enumerate(comps):
        bands = cc.dwt_bands(img[w0:w1])
        for ri, bi in enumerate(bidx):
            b = bands[bi]
            vmax = max(float(np.abs(b).max()), 5.0)
            im = _coeff_map(ax[ri, ci], b, f'{cname} — band {labels[bi]}', nblocks, vmax)
            fig.colorbar(im, ax=ax[ri, ci], fraction=0.046, pad=0.04)
    fig.suptitle(f'{ptype} plane — coefficient maps (wire x coeff, {cc.WAVELET} L{cc.LEVEL}): '
                 f'coherent is CONSTANT within each 64-block (vertical stripes between black lines),\n'
                 f'intrinsic is per-wire speckle, signal is sparse streaks. Wires {w0}-{w1}.',
                 fontweight='bold', fontsize=10.5)
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, f'fig4_coeff_maps_{ptype}.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print('saved', p)


# ---------------------------------------------------------------- Fig 5
def fig_level_energy(signal, coherent, intrinsic, ptype):
    labels = cc.band_labels()
    comps = [('signal', signal, 'tab:orange'), ('coherent', coherent, 'tab:red'),
             ('intrinsic', intrinsic, 'tab:blue')]
    energies, madsigs = {}, {}
    for cname, img, _ in comps:
        bands = cc.dwt_bands(img)
        e = np.array([float(np.sum(b.astype(np.float64) ** 2)) for b in bands])
        energies[cname] = 100 * e / e.sum()
        madsigs[cname] = np.array([float(np.median(np.abs(b)) / 0.6745) for b in bands])
    x = np.arange(len(labels)); wbar = 0.26
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for k, (cname, _, col) in enumerate(comps):
        ax[0].bar(x + (k - 1) * wbar, energies[cname], wbar, label=cname, color=col)
        ax[1].bar(x + (k - 1) * wbar, madsigs[cname], wbar, label=cname, color=col)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].set_ylabel('energy %')
    ax[0].set_title('energy fraction per level'); ax[0].legend()
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_ylabel('MAD sigma (ADC)')
    ax[1].set_title('per-band noise scale (MAD sigma)'); ax[1].legend()
    fig.suptitle(f'{ptype} plane — where each component lives in scale space',
                 fontweight='bold')
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, f'fig5_level_energy_{ptype}.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print('saved', p)


# ---------------------------------------------------------------- Fig 6
def fig_cross_block(coherent, ptype):
    nw = coherent.shape[0]
    ng = cc.full_groups(nw)
    flat = cc.flat_coeffs(cc.dwt_bands(coherent))
    reps = np.stack([flat[g * GS] for g in range(ng)])
    reps = reps - reps.mean(1, keepdims=True)
    reps /= np.maximum(np.linalg.norm(reps, axis=1, keepdims=True), 1e-30)
    C = reps @ reps.T
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    im = ax[0].imshow(C, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax[0].set_title('block-to-block coherent corr (coeff space)')
    ax[0].set_xlabel('block'); ax[0].set_ylabel('block')
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    lags = range(0, 6)
    means = [np.mean([C[g, g + d] for g in range(ng - d)]) for d in lags]
    stds = [np.std([C[g, g + d] for g in range(ng - d)]) for d in lags]
    ax[1].errorbar(list(lags), means, yerr=stds, marker='o', capsize=3)
    ax[1].axhline(0, color='k', lw=0.5)
    beta = 0.15
    ax[1].axhline(-2 * beta / (1 + 2 * beta ** 2), color='tab:red', ls='--', lw=1,
                  label=f'model lag1 = -2b/(1+2b^2) = {-2*beta/(1+2*beta**2):.2f}')
    ax[1].set_xlabel('block lag'); ax[1].set_ylabel('mean corr')
    ax[1].set_title('coherent "bleeding" decays after 1 neighbor'); ax[1].legend(fontsize=8)
    fig.suptitle(f'{ptype} plane — cross-block coherent correlation (beta=0.15)',
                 fontweight='bold')
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, f'fig6_cross_block_{ptype}.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print('saved', p)


def main():
    ptype = sys.argv[1] if len(sys.argv) > 1 else 'Y'
    event = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    signal, coherent, intrinsic = cc.components(ptype, event)
    print(f'=== figures: plane {ptype}, event {event} ===')
    fig_within_vs_across(coherent, ptype)
    fig_signal_adjacent(signal, coherent, ptype)
    fig_decomp_levels(signal, coherent, intrinsic, ptype)
    fig_coeff_maps(signal, coherent, intrinsic, ptype)
    fig_level_energy(signal, coherent, intrinsic, ptype)
    fig_cross_block(coherent, ptype)


if __name__ == '__main__':
    main()
