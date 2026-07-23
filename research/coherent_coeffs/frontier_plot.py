"""Plot the F0-vs-coefficient frontier from frontier.json (smart k-curve, helix, oracle)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

J = json.load(open(os.path.join(cc.FIGDIR, '..', 'frontier.json')))
planes = ['Y', 'U', 'V']
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
for j, p in enumerate(planes):
    d = J[p]
    sk = np.array(d['smart_kept']) / 1000; sf = np.array(d['smart_f0'])
    ax[j].plot(sk, sf, 'o-', color='tab:red', label='smart (k-sweep)', zorder=3)
    for k, x, y in zip(d['ks'], sk, sf):
        ax[j].annotate(f'k{k:g}', (x, y), fontsize=7, textcoords='offset points', xytext=(3, 4))
    ax[j].scatter([d['helix_kept'] / 1000], [d['helix_f0']], color='tab:gray', s=90,
                  marker='s', label='helix', zorder=4)
    ax[j].scatter([d['oracle_kept'] / 1000], [d['oracle_f0']], color='tab:green', s=110,
                  marker='*', label='oracle (no coherent)', zorder=5)
    ax[j].axhline(d['oracle_f0'], color='tab:green', ls=':', lw=1, alpha=0.7)
    ax[j].axvline(d['oracle_kept'] / 1000, color='tab:green', ls=':', lw=1, alpha=0.7)
    ax[j].set_title(f'plane {p}'); ax[j].set_xlabel('kept coeffs / plane (x1000)')
    ax[j].set_ylabel('F0'); ax[j].legend(fontsize=8)
fig.suptitle('F0 vs coefficients (30 events): smart reaches the no-coherent ORACLE count; '
             'F0 peaks at k~3 then gate eats signal.\nLower-right = better. smart dominates helix.',
             fontweight='bold', fontsize=11)
fig.tight_layout()
pth = os.path.join(cc.FIGDIR, 'fig12_frontier.png')
fig.savefig(pth, dpi=130); plt.close(fig)
print('saved', pth)
