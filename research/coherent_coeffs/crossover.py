"""Centerpiece figure: per-level separability of coherent vs signal common-mode.

For each plane, deep decomposition (sym4 L9), averaged over events:
  left  : signal-common-mode / coherent-common-mode  RATIO per level, naive vs
          k-sigma-masked. Ratio>1 (shaded) = signal swamps coherent -> coefficient
          estimation of coherent fails. Shows the crossover scale per plane and
          that going DEEPER (more coarse bands) does not help (coarse ratios stay high).
  right : the two universal facts — within-block coeff deviation (==0 at all levels)
          and cross-block lag-1 correlation (~-0.29, the beta bleeding).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cc_common as cc
GS = cc.GROUP_SIZE


def per_level_ratios(ptype, level, wavelet, events, ksig=3.0):
    nb = level + 1
    accum = {k: np.zeros(nb) for k in ('coh', 'naive', 'mask', 'wdev', 'lag1')}
    for e in events:
        signal, coherent, _ = cc.components(ptype, e)
        nw = signal.shape[0]; ngf = cc.full_groups(nw)
        sb = cc.dwt_bands(signal, wavelet, level)
        cb = cc.dwt_bands(coherent, wavelet, level)
        for bi in range(nb):
            s, c = sb[bi], cb[bi]
            L = c.shape[-1]
            # coherent common-mode rms (block coeff value)
            reps = np.stack([c[g * GS] for g in range(ngf)])
            accum['coh'][bi] += np.sqrt(np.mean(reps ** 2))
            # signal common-mode naive / masked
            blk = s[:ngf * GS].reshape(ngf, GS, L)
            accum['naive'][bi] += np.sqrt(np.mean(blk.mean(1) ** 2))
            med = np.median(blk, axis=1, keepdims=True)
            r = blk - med
            sg = np.maximum(np.median(np.abs(r), axis=(1, 2)) / 0.6745, 1e-6)[:, None, None]
            uf = np.abs(r) <= ksig * sg
            mm = (blk * uf).sum(1) / np.maximum(uf.sum(1), 1)
            accum['mask'][bi] += np.sqrt(np.mean(mm ** 2))
            # within-block deviation (coherent)
            wd = max(float(np.max(np.abs(cb_blk - cb_blk[0:1])))
                     for cb_blk in (c[g * GS:(g + 1) * GS] for g in range(ngf)))
            accum['wdev'][bi] += wd
            rn = reps - reps.mean(1, keepdims=True)
            rn /= np.maximum(np.linalg.norm(rn, axis=1, keepdims=True), 1e-30)
            C = rn @ rn.T
            accum['lag1'][bi] += np.mean([C[g, g + 1] for g in range(ngf - 1)])
    for k in accum:
        accum[k] /= len(events)
    return accum


def main():
    level, wavelet = 9, 'sym4'
    events = list(range(0, 4 * 37, 37))
    labels = cc.band_labels(level)
    x = np.arange(len(labels))
    planes = ['Y', 'U', 'V']
    cols = {'Y': 'tab:orange', 'U': 'tab:green', 'V': 'tab:blue'}
    data = {p: per_level_ratios(p, level, wavelet, events) for p in planes}

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].axhspan(1.0, 6, color='red', alpha=0.07)
    ax[0].axhline(1.0, color='red', lw=1, ls='--', label='signal = coherent')
    for p in planes:
        d = data[p]
        ax[0].plot(x, d['naive'] / np.maximum(d['coh'], 1e-9), 'o-', color=cols[p],
                   label=f'{p} naive')
        ax[0].plot(x, d['mask'] / np.maximum(d['coh'], 1e-9), 's--', color=cols[p],
                   alpha=0.6, label=f'{p} k-sig masked')
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_xlabel('level (coarse - > fine)')
    ax[0].set_ylabel('signal common-mode / coherent common-mode')
    ax[0].set_title('separability per level: ratio>1 (shaded) = signal swamps coherent\n'
                    'deeper levels add MORE swamped coarse bands — they do not help', fontsize=10)
    ax[0].set_ylim(0, 5); ax[0].legend(fontsize=7, ncol=3, loc='upper right')

    # right: universal facts
    ax2 = ax[1]
    for p in planes:
        ax2.plot(x, data[p]['lag1'], 'o-', color=cols[p], label=f'{p} cross-block lag1 corr')
    ax2.axhline(-0.287, color='k', ls=':', lw=1, label='beta model -0.29')
    ax2.axhline(0.0, color='gray', lw=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_xlabel('level (coarse - > fine)'); ax2.set_ylabel('correlation')
    wdev_max = max(float(np.max(data[p]['wdev'])) for p in planes)
    ax2.set_title(f'universal at ALL levels: within-block coeff deviation = {wdev_max:.1e}\n'
                  f'(coherent identical within a block); neighbors anti-correlated', fontsize=10)
    ax2.legend(fontsize=8, loc='lower right'); ax2.set_ylim(-0.45, 0.1)

    fig.suptitle(f'Coherent vs signal in wavelet space — per level ({wavelet} L{level}, '
                 f'{len(events)} events)', fontweight='bold')
    fig.tight_layout()
    p = os.path.join(cc.FIGDIR, 'fig7_crossover_per_level.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    print('saved', p)
    # print the crossover band per plane
    for pl in planes:
        d = data[pl]; r = d['naive'] / np.maximum(d['coh'], 1e-9)
        below = [labels[i] for i in range(len(labels)) if r[i] < 1.0]
        print(f'  {pl}: ratio<1 (coherent recoverable, naive) at levels {below}')


if __name__ == '__main__':
    main()
