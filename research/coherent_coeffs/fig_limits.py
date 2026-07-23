"""Measured limit decomposition (12 events): ours (de3c) vs mask-ceiling (oracle_pt) vs
true ceiling (no-coherent). Two gaps: DETECTION (recoverable, hard) + estimation FLOOR (fundamental)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

D = {  # de3c, oracle_pt, nocoh  (12 ev)
    'Y': (0.9613, 0.9645, 0.9667), 'U': (0.8917, 0.9201, 0.9317), 'V': (0.8951, 0.9075, 0.9119)}
planes = list(D); x = np.arange(3)
fig, ax = plt.subplots(figsize=(10, 5.5))
ours = [D[p][0] for p in planes]; mo = [D[p][1] for p in planes]; nc = [D[p][2] for p in planes]
ax.bar(x, ours, 0.5, color='tab:red', label='de3c (achieved)')
ax.bar(x, [mo[i]-ours[i] for i in range(3)], 0.5, bottom=ours, color='tab:orange', alpha=0.7,
       label='DETECTION gap (recoverable, hard)')
ax.bar(x, [nc[i]-mo[i] for i in range(3)], 0.5, bottom=mo, color='tab:gray', alpha=0.7,
       label='estimation FLOOR gap (fundamental)')
for i, p in enumerate(planes):
    ax.text(i, ours[i]-0.004, f'{ours[i]:.3f}', ha='center', color='white', fontsize=9)
    ax.text(i, mo[i]+0.0005, f'det {mo[i]-ours[i]:+.3f}', ha='center', fontsize=8)
    ax.text(i, nc[i]+0.0005, f'floor {nc[i]-mo[i]:+.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f'{p}' for p in planes]); ax.set_ylim(0.86, 0.975)
ax.set_ylabel('F0_recon'); ax.legend(loc='lower right', fontsize=9)
ax.set_title('How close to the limit (12 ev): de3c + the two measured gaps.\nDetection gap = dense-region '
             'clean-wire ID at SNR~1 (hard); Floor gap = injected sigma_int/sqrt(n_clean) (fundamental)',
             fontsize=10, fontweight='bold')
fig.tight_layout(); p = os.path.join(cc.FIGDIR, 'fig20_limits.png'); fig.savefig(p, dpi=130); plt.close(fig)
print('saved', p)
