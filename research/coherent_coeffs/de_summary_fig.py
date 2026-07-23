"""Summary bars: helix / smart / de / oracle, F0_recon + coh_left per plane (10 events)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

# from de_metrics.py (12 events, de = ksig 1.5): F0_recon, coh_left
D = {
    'Y': {'helix': (0.9578, 0.560), 'smart': (0.9587, 0.384), 'de': (0.9638, 0.261), 'oracle': (0.9658, 0.0)},
    'U': {'helix': (0.8956, 0.784), 'smart': (0.8864, 0.537), 'de': (0.8866, 0.552), 'oracle': (0.9333, 0.0)},
    'V': {'helix': (0.8836, 0.714), 'smart': (0.8855, 0.357), 'de': (0.8898, 0.336), 'oracle': (0.9120, 0.0)},
}
planes = ['Y', 'U', 'V']; methods = ['helix', 'smart', 'de', 'oracle']
cols = {'helix': 'tab:gray', 'smart': 'tab:blue', 'de': 'tab:red', 'oracle': 'tab:green'}
x = np.arange(len(planes)); w = 0.2
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for j, m in enumerate(methods):
    ax[0].bar(x + (j - 1.5) * w, [D[p][m][0] for p in planes], w, label=m, color=cols[m])
    ax[1].bar(x + (j - 1.5) * w, [D[p][m][1] for p in planes], w, label=m, color=cols[m])
ax[0].set_xticks(x); ax[0].set_xticklabels(planes); ax[0].set_ylim(0.84, 0.97)
ax[0].set_title('reconstruction F0 (higher better)'); ax[0].set_ylabel('F0_recon'); ax[0].legend()
ax[1].set_xticks(x); ax[1].set_xticklabels(planes)
ax[1].set_title('leftover coherent RMS (lower better)'); ax[1].set_ylabel('coh_left ADC'); ax[1].legend()
fig.suptitle('detect-then-estimate (de, ksig 1.5) vs smart vs oracle (12 events): de >= smart on '
             'every plane; reaches oracle on Y; U headroom stays open', fontweight='bold')
fig.tight_layout()
p = os.path.join(cc.FIGDIR, 'fig16_de_summary.png')
fig.savefig(p, dpi=130); plt.close(fig)
print('saved', p)
