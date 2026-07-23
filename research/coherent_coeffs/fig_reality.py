"""Ground the F0 differences in reality: 1-F0 as % signal-charge error, decomposed into
irreducible (intrinsic+sparsify floor) vs coherent residual, with per-event scatter as error
bars. Shows the de2-vs-smart and tuning gains are small vs both the floor and the scatter."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

# 20-ev: (1-F0)*100 = % charge error; (mean, std)
D = {'Y': {'smart': (4.18, 0.49), 'de2cl': (3.92, 0.40), 'floor': (3.41, 0)},
     'U': {'smart': (11.03, 1.41), 'de2cl': (10.00, 1.26), 'floor': (6.48, 0)},
     'V': {'smart': (11.09, 1.30), 'de2cl': (10.53, 1.34), 'floor': (8.54, 0)}}
planes = ['Y', 'U', 'V']; x = np.arange(3); w = 0.28
fig, ax = plt.subplots(figsize=(11, 5.5))
for j, (m, col) in enumerate([('smart', 'tab:blue'), ('de2cl', 'tab:red'), ('floor', 'tab:green')]):
    vals = [D[p][m][0] for p in planes]; err = [D[p][m][1] for p in planes]
    ax.bar(x + (j - 1) * w, vals, w, yerr=err, capsize=4, color=col,
           label={'smart': 'smart', 'de2cl': 'de2_clamp', 'floor': 'irreducible floor (no coherent)'}[m])
ax.axhspan(0, 5, color='gray', alpha=0.08)
ax.text(2.35, 4.6, 'typical LArTPC\ncharge resolution\n(several %)', fontsize=8, color='gray', ha='center')
ax.set_xticks(x); ax.set_xticklabels(['Y (sig~49 ADC)', 'U (~20)', 'V (~18)'])
ax.set_ylabel('signal-charge error  1-F0  (%)'); ax.legend()
ax.set_title('Grounding the F0 gaps: 1-F0 = % of signal charge mis-reconstructed.\n'
             'de2_clamp shaves ~1% off smart (U) — WITHIN the per-event scatter (error bars) and below\n'
             'detector resolution. Most of the error is the irreducible intrinsic+sparsify floor (green).',
             fontsize=10, fontweight='bold')
fig.tight_layout(); pth = os.path.join(cc.FIGDIR, 'fig27_reality.png')
fig.savefig(pth, dpi=130); plt.close(fig); print('saved', pth)
