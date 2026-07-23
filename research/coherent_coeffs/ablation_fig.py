"""Build-up ablation across planes: F0 gain vs smart at each ladder step, per plane.
Shows what each algorithm component buys for collection (Y) vs induction (U,V)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

STEPS = ['smart', 'helix', 'amp1', 'amp4', 'hyst4', 'hyst4_d15', '+clamp', '+joint']
F = {  # 12-ev F0_recon
 'Y': [0.9586, 0.9586, 0.9623, 0.9631, 0.9634, 0.9638, 0.9635, 0.9633],
 'U': [0.8828, 0.8946, 0.8812, 0.8667, 0.8818, 0.8879, 0.8926, 0.8926],
 'V': [0.8890, 0.8872, 0.8901, 0.8880, 0.8916, 0.8973, 0.8975, 0.8974]}
cols = {'Y': 'tab:orange', 'U': 'tab:green', 'V': 'tab:blue'}
x = np.arange(len(STEPS))
fig, ax = plt.subplots(figsize=(12, 5.5))
for p in F:
    d = np.array(F[p]) - F[p][0]   # gain vs smart
    ax.plot(x, 1000 * d, 'o-', color=cols[p], label=f'{p} (smart F0={F[p][0]:.3f})')
ax.axhline(0, color='k', lw=0.5); ax.set_xticks(x); ax.set_xticklabels(STEPS, rotation=20)
ax.set_ylabel('F0 gain vs smart (x1000)')
ax.set_title('Build-up ablation across planes (12 ev): what each component buys.\n'
             'Y(collection): simple detect ~optimal, extras neutral. U/V(induction): PLAIN amp HURTS '
             '(-13/-15 mU), HYSTERESIS essential, dilation+clamp help U. JOINT ~0 everywhere.',
             fontsize=10, fontweight='bold')
ax.legend(); ax.grid(alpha=0.2)
fig.tight_layout(); pth = os.path.join(cc.FIGDIR, 'fig24_ablation_buildup.png')
fig.savefig(pth, dpi=130); plt.close(fig); print('saved', pth)
