"""Ablation summary: F0 gain vs smart per plane for a ladder of methods (15 events).
Shows the simplest near-best per plane and that joint adds nothing."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

# gains vs smart (ablate2.log i4 + ablate.log de3_clamp), 15 events
G = {  # method: {plane: gain}
    'plain-thr (simplest)':      {'Y': 0.0059, 'U': 0.0002, 'V': 0.0030},
    'plain-thr + clamp':         {'Y': 0.0054, 'U': 0.0050, 'V': 0.0037},
    'hysteresis + clamp':        {'Y': 0.0035, 'U': 0.0099, 'V': 0.0053},
    'hysteresis + clamp + joint':{'Y': 0.0036, 'U': 0.0104, 'V': 0.0055},
}
planes = ['Y', 'U', 'V']; methods = list(G)
cols = ['tab:gray', 'tab:cyan', 'tab:red', 'tab:purple']
x = np.arange(3); w = 0.2
fig, ax = plt.subplots(figsize=(11, 5))
for j, m in enumerate(methods):
    ax.bar(x + (j - 1.5) * w, [1000 * G[m][p] for p in planes], w, label=m, color=cols[j])
ax.set_xticks(x); ax.set_xticklabels(['Y (collection)', 'U (induction)', 'V (induction)'])
ax.set_ylabel('F0 gain vs smart (x1000)'); ax.axhline(0, color='k', lw=0.5)
ax.set_title('Ablation (15 ev): simplest plain-threshold best for COLLECTION; hysteresis+clamp '
             'best for INDUCTION;\njoint step adds ~nothing (drop it). Clamp essential for induction.',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=9)
fig.tight_layout(); p = os.path.join(cc.FIGDIR, 'fig19_ablation.png')
fig.savefig(p, dpi=130); plt.close(fig); print('saved', p)
