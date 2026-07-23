"""Summary: de2 (induction-capable detect-then-estimate) vs smart vs oracle, + the
fundamental-gap diagnostic (error vs block occupancy / dense-run lengths)."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc
import induction as ind

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'wire_denoise'))
import common as wd  # noqa: E402
GS = cc.GROUP_SIZE

# 30-event F0_recon / coh_left (validate_clamp.py) — FINAL method de3_clamp
D = {'Y': {'smart': (0.9567, 0.430), 'final': (0.9603, 0.338), 'oracle': (0.9643, 0.238)},
     'U': {'smart': (0.8874, 0.611), 'final': (0.8981, 0.527), 'oracle': (0.9201, 0.359)},
     'V': {'smart': (0.8887, 0.376), 'final': (0.8953, 0.341), 'oracle': (0.9069, 0.257)}}


def fig_summary():
    planes = ['Y', 'U', 'V']; methods = ['smart', 'final', 'oracle']
    cols = {'smart': 'tab:blue', 'final': 'tab:red', 'oracle': 'tab:green'}
    x = np.arange(3); w = 0.25
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, m in enumerate(methods):
        ax[0].bar(x + (j - 1) * w, [D[p][m][0] for p in planes], w, label=m, color=cols[m])
        ax[1].bar(x + (j - 1) * w, [D[p][m][1] for p in planes], w, label=m, color=cols[m])
    ax[0].set_xticks(x); ax[0].set_xticklabels(planes); ax[0].set_ylim(0.86, 0.97)
    ax[0].set_title('reconstruction F0 (higher better)'); ax[0].legend()
    ax[1].set_xticks(x); ax[1].set_xticklabels(planes)
    ax[1].set_title('leftover coherent RMS (lower better)'); ax[1].legend()
    fig.suptitle('FINAL coherent removal (de3-clamp) vs smart vs oracle (30 events): >= smart all '
                 'planes, per-event safe; Y at oracle, V within 0.012, U within 0.022 (fundamental '
                 'signal-majority long-track limit)', fontweight='bold', fontsize=10)
    fig.tight_layout(); p = os.path.join(cc.FIGDIR, 'fig17_induction_summary.png')
    fig.savefig(p, dpi=130); plt.close(fig); print('saved', p)


def fig_gap():
    """U: per-block estimate error vs occupancy (de2 vs oracle) + dense-run length hist."""
    s, c, i = cc.components('U', 7); noisy = wd.digitize(s + c + i, cc.PLANES['U']['pedestal'])
    coh, _ = ind.iterate(noisy, n_iter=4, klo=0.7, dilate=15, seed='amp')
    coh_or = ind.estimate(noisy, ind.mask_true(s))
    ng = cc.full_groups(noisy.shape[0])
    occ = np.array([(np.abs(s[g * GS:(g + 1) * GS]) > 5).sum(0).max() for g in range(ng)])
    e_ours = np.array([np.sqrt(np.mean((coh[g * GS] - c[g * GS]) ** 2)) for g in range(ng)])
    e_or = np.array([np.sqrt(np.mean((coh_or[g * GS] - c[g * GS]) ** 2)) for g in range(ng)])
    runs = []
    for g in range(ng):
        oc = (np.abs(s[g * GS:(g + 1) * GS]) > 5).sum(0) > 32
        d = np.diff(np.concatenate([[0], oc.astype(int), [0]]))
        runs.extend((np.where(d == -1)[0] - np.where(d == 1)[0]).tolist())
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].scatter(occ, e_ours, c='tab:red', label='de2', s=30)
    ax[0].scatter(occ, e_or, c='tab:green', label='oracle (true mask)', s=30, marker='x')
    ax[0].axvline(32, color='k', ls=':', lw=1); ax[0].set_xlabel('block max occupancy (#wires)')
    ax[0].set_ylabel('per-block coherent error (ADC)'); ax[0].legend()
    ax[0].set_title('U: error rises with occupancy; gap is in DENSE blocks (occ>32)')
    ax[1].hist(runs, bins=20, color='tab:gray', edgecolor='k')
    ax[1].axvline(50, color='tab:red', ls='--', label='~coherent corr length')
    ax[1].set_xlabel('dense-run length (ticks)'); ax[1].set_ylabel('count'); ax[1].legend()
    ax[1].set_title('dense runs: long ones (>50t) = parallel tracks, unrecoverable')
    fig.suptitle('U fundamental gap: long parallel-track runs have no clean-wire majority '
                 'and no temporal anchor', fontweight='bold')
    fig.tight_layout(); p = os.path.join(cc.FIGDIR, 'fig18_induction_gap.png')
    fig.savefig(p, dpi=130); plt.close(fig); print('saved', p)


if __name__ == '__main__':
    fig_summary(); fig_gap()
