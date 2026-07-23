"""Valid F0-vs-coeff frontier: standard vs hysteresis sparsify (same de3c image), U/V."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

# actual frontier_compare 10ev output
STD = {'U': [(34, .8797), (37, .8875), (45, .8945), (65, .8988), (139, .9016)],
       'V': [(27, .8763), (30, .8864), (38, .8962), (58, .9027), (132, .9080)]}
HYS = {'U': [(46, .8960), (51, .9002), (57, .9035), (66, .9049), (87, .9056)],
       'V': [(38, .8983), (42, .9043), (49, .9096), (58, .9122), (79, .9139)]}
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for j, p in enumerate(['U', 'V']):
    sk = [x[0] for x in STD[p]]; sf = [x[1] for x in STD[p]]
    hk = [x[0] for x in HYS[p]]; hf = [x[1] for x in HYS[p]]
    ax[j].plot(sk, sf, 'o-', color='tab:gray', label='standard sparsify (kappa-sweep) = PRIOR')
    ax[j].plot(hk, hf, 's-', color='tab:red', label='hysteresis sparsify (grow-sweep)')
    ax[j].scatter([STD[p][2][0]], [STD[p][2][1]], color='k', zorder=5, s=60,
                  label='production (kappa=1)')
    ax[j].set_xlabel('kept coeffs / plane (x1000)'); ax[j].set_ylabel('F0_recon')
    ax[j].set_title(f'{p}: de3c-cleaned'); ax[j].legend(fontsize=8)
fig.suptitle('VALID comparison (same image, matched coeffs): hysteresis frontier is above standard.\n'
             'Standard saturates (kept noise reconstructs onto signal); hysteresis keeps climbing. '
             'Gain at production budget ~+0.001; up to +0.005-0.008 if spending more coeffs.',
             fontweight='bold', fontsize=9.5)
fig.tight_layout(); pth = os.path.join(cc.FIGDIR, 'fig23_frontier_compare.png')
fig.savefig(pth, dpi=130); plt.close(fig); print('saved', pth)
