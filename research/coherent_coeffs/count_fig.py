"""Coefficient-count figure from the 50-event run (count.py).
Left: kept coeffs/plane (lower=better) raw/helix/smart@matched-F0.
Right: F0 (higher=better). Smart keeps fewer coeffs at >= helix F0 on Y/V, tie on U.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

# 50-event results (count.py). smart = best-k at matched/better F0.
D = {  # plane: (raw_kept, helix_kept, smart_kept, raw_f0, helix_f0, smart_f0, smart_k)
    'Y': (35352, 50543, 47371, 0.9341, 0.9566, 0.9570, 4.0),
    'U': (35905, 62776, 61949, 0.8428, 0.8972, 0.8968, 3.0),
    'V': (30829, 53818, 46276, 0.8309, 0.8835, 0.8877, 3.5),
}
planes = list(D)
x = np.arange(len(planes)); w = 0.26
cols = {'raw': 'lightgray', 'helix': 'tab:gray', 'smart': 'tab:red'}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for j, m in enumerate(['raw', 'helix', 'smart']):
    kept = [D[p][{'raw': 0, 'helix': 1, 'smart': 2}[m]] / 1000 for p in planes]
    ax[0].bar(x + (j - 1) * w, kept, w, label=m, color=cols[m])
for j, m in enumerate(['raw', 'helix', 'smart']):
    f0 = [D[p][{'raw': 3, 'helix': 4, 'smart': 5}[m]] for p in planes]
    ax[1].bar(x + (j - 1) * w, f0, w, label=m, color=cols[m])
ax[0].set_xticks(x); ax[0].set_xticklabels(planes); ax[0].set_ylabel('kept coeffs / plane (x1000)')
ax[0].set_title('coefficients kept after sparsify (lower = better)'); ax[0].legend()
for p in planes:  # annotate smart vs helix delta
    i = planes.index(p)
    d = 100 * (D[p][2] - D[p][1]) / D[p][1]
    ax[0].text(i + w, D[p][2] / 1000, f'{d:+.0f}%', ha='center', va='bottom', fontsize=9, color='tab:red')
ax[1].set_xticks(x); ax[1].set_xticklabels(planes); ax[1].set_ylabel('F0'); ax[1].set_ylim(0.82, 0.97)
ax[1].set_title('reconstruction F0 (higher = better)'); ax[1].legend()
fig.suptitle('End-to-end: smart level-aware coherent removal -> sparsify (50 events, coif3 L4)\n'
             'smart keeps fewer coeffs at >= helix F0 (Y -6%, V -14%; U tie)', fontweight='bold')
fig.tight_layout()
p = os.path.join(cc.FIGDIR, 'fig11_coeff_count.png')
fig.savefig(p, dpi=130); plt.close(fig)
print('saved', p)
