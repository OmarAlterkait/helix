"""Per-plane: noise INSIDE vs OUTSIDE signal for key methods (the real discriminator).
nz_out hits the floor (~sigma_int) for all removers; nz_in (on-signal coherent residual) differs."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

# 12-ev (ablation_metrics): method -> (F0, kept, nz_in, nz_out)
D = {
 'Y': {'raw': (.9370, 30009, 3.01, 3.01), 'smart': (.9586, 40200, 2.32, 1.62),
       'helix': (.9586, 43112, 2.32, 1.67), 'de2_clamp': (.9635, 39941, 1.93, 1.60)},
 'U': {'raw': (.8397, 30490, 3.06, 3.04), 'smart': (.8828, 45212, 3.01, 1.67),
       'helix': (.8946, 55489, 2.72, 1.79), 'de2_clamp': (.8926, 45051, 2.83, 1.66)},
 'V': {'raw': (.8360, 25267, 3.07, 3.04), 'smart': (.8890, 37250, 2.38, 1.66),
       'helix': (.8872, 46461, 2.50, 1.78), 'de2_clamp': (.8975, 36784, 2.15, 1.66)}}
methods = ['smart', 'helix', 'de2_clamp']; cols = ['tab:blue', 'tab:gray', 'tab:red']
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
x = np.arange(len(methods)); w = 0.35
for j, p in enumerate(['Y', 'U', 'V']):
    nin = [D[p][m][2] for m in methods]; nout = [D[p][m][3] for m in methods]
    ax[j].bar(x - w / 2, nin, w, color='tab:red', label='nz_in (on signal)')
    ax[j].bar(x + w / 2, nout, w, color='tab:cyan', label='nz_out (off signal)')
    ax[j].axhline(1.6, color='k', ls=':', lw=1, label='intrinsic floor ~1.6')
    for k, m in enumerate(methods):
        ax[j].text(k, max(nin[k], nout[k]) + 0.05, f'F0 {D[p][m][0]:.3f}\n{D[p][m][1]/1000:.0f}k',
                   ha='center', fontsize=7)
    ax[j].set_xticks(x); ax[j].set_xticklabels(methods, rotation=15); ax[j].set_ylim(0, 3.4)
    ax[j].set_title(f'{p}'); ax[j].set_ylabel('RMS residual (ADC)')
    if j == 0:
        ax[j].legend(fontsize=8)
fig.suptitle('Noise inside vs outside signal (12 ev). OFF-signal coherent removal is SOLVED '
             '(nz_out~floor for all); the ON-signal residual (nz_in) is the hard discriminator.\n'
             'Y/V: de2_clamp lowest nz_in + fewest coeffs. U: helix edges nz_in (2.72) but +22% coeffs '
             '& worse nz_out; smart leaves ~all coherent on U signal (3.01).', fontsize=9.5, fontweight='bold')
fig.tight_layout(); pth = os.path.join(cc.FIGDIR, 'fig25_noise_in_out.png')
fig.savefig(pth, dpi=130); plt.close(fig); print('saved', pth)
