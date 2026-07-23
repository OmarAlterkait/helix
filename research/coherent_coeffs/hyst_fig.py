"""F0-vs-kept: hysteresis sparsify (local keep-near-signal) dominates the global-kappa knob."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

# global-kappa std curve (kappa_knob 10ev) and hysteresis points (hyst_sparsify 8ev)
STD = {'U': [(36273, .8857), (45416, .8945), (80347, .9000), (279302, .9021), (1187212, .8993)],
       'V': [(28675, .8839), (37526, .8962), (72614, .9047), (272949, .9103), (1184093, .9107)]}
# actual measured (hyst_sparsify 8ev): grow 0.5, 0.35
HYST = {'U': [(70520, .9077), (171423, .9071)],
        'V': [(64559, .9094), (169706, .9108)]}
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for j, p in enumerate(['U', 'V']):
    sk = [x[0] / 1000 for x in STD[p]]; sf = [x[1] for x in STD[p]]
    hk = [x[0] / 1000 for x in HYST[p]]; hf = [x[1] for x in HYST[p]]
    ax[j].plot(sk, sf, 'o-', color='tab:gray', label='global kappa (keep everywhere)')
    ax[j].plot(hk, hf, 's-', color='tab:red', label='hysteresis (keep near signal only)')
    for kg, x, y in zip([0.5, 0.35], hk, hf):
        ax[j].annotate(f'grow{kg}', (x, y), fontsize=7, textcoords='offset points', xytext=(4, -8))
    ax[j].set_xscale('log'); ax[j].set_xlabel('kept coeffs / plane (x1000, log)')
    ax[j].set_ylabel('F0_recon'); ax[j].set_title(f'{p}'); ax[j].legend(fontsize=8)
fig.suptitle('Local "keep more near signal" knob (wavelet hysteresis) DOMINATES global-kappa:\n'
             'same F0 recovery at ~4-10x fewer extra coefficients', fontweight='bold', fontsize=10)
fig.tight_layout(); pth = os.path.join(cc.FIGDIR, 'fig22_hysteresis_sparsify.png')
fig.savefig(pth, dpi=130); plt.close(fig); print('saved', pth)
