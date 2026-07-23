"""F0-vs-kept frontier (the kappa knob): de3c vs oracle, U/V. Shows keeping more coeffs
(lower kappa) recovers F0 — reaching the oracle's production F0 for V, plateauing for U."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

K = [0.4, 0.6, 0.8, 1.0, 1.25]
D = {
 'U': {'de3c': [(1187212, .8993), (279302, .9021), (80347, .9000), (45416, .8945), (36273, .8857)],
       'oracle': [(1190461, .9210), (283452, .9254), (82528, .9241), (46387, .9187), (37145, .9090)]},
 'V': {'de3c': [(1184093, .9107), (272949, .9103), (72614, .9047), (37526, .8962), (28675, .8839)],
       'oracle': [(1185087, .9220), (274101, .9218), (72886, .9160), (37381, .9067), (28813, .8936)]},
}
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for j, p in enumerate(['U', 'V']):
    for name, col in [('de3c', 'tab:red'), ('oracle', 'tab:green')]:
        pts = D[p][name]; kept = [x[0] / 1000 for x in pts]; f0 = [x[1] for x in pts]
        ax[j].plot(kept, f0, 'o-', color=col, label=name)
        for k, x, y in zip(K, kept, f0):
            ax[j].annotate(f'k{k:g}', (x, y), fontsize=7, textcoords='offset points', xytext=(3, 3))
    # production point (kappa=1) horizontal ref = oracle@k1
    ax[j].axhline(D[p]['oracle'][3][1], color='tab:green', ls=':', lw=1, alpha=0.7,
                  label='oracle F0 @k1 (target)')
    ax[j].set_xscale('log'); ax[j].set_xlabel('kept coeffs / plane (x1000, log)'); ax[j].set_ylabel('F0_recon')
    ax[j].set_title(f'{p}: lower kappa = keep more noise -> higher F0'); ax[j].legend(fontsize=8)
fig.suptitle('The kappa knob (rate-distortion): keep more coeffs/noise to raise F0. V reaches the '
             'oracle F0 (~2x coeffs); U plateaus below it (residual coherent on signal caps F0).',
             fontweight='bold', fontsize=10)
fig.tight_layout(); pth = os.path.join(cc.FIGDIR, 'fig21_kappa_knob.png'); fig.savefig(pth, dpi=130); plt.close(fig)
print('saved', pth)
