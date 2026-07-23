"""Knob sensitivity: F0 range (max-min over each knob's sweep) per plane.
Bigger bar = knob matters more. From sweep_knob 10-ev results."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

# F0 range (max-min) over each knob's sweep, per plane (1000x). klo excludes the percolating 0.5.
RANGE = {  # knob: (Y, U, V)  in milli-F0
 'kgate':    (0.7, 16.0, 0.6),
 'detector': (2.8, 8.6, 3.2),
 'dilate':   (3.2, 9.4, 4.2),
 'baseline': (0.6, 8.6, 2.4),
 'n_iter':   (0.8, 7.8, 4.2),
 'clamp':    (0.4, 3.6, 1.3),
 'klo*':     (2.3, 2.2, 1.0),   # *excluding klo<=0.5 which percolates
 'minc':     (1.7, 1.5, 1.0),
 'khi':      (0.2, 1.4, 0.5),
 'reducer':  (0.2, 0.5, 0.4)}
knobs = list(RANGE); x = np.arange(len(knobs)); w = 0.27
cols = {'Y': 'tab:orange', 'U': 'tab:green', 'V': 'tab:blue'}
fig, ax = plt.subplots(figsize=(12, 5.2))
for j, p in enumerate(['Y', 'U', 'V']):
    ax.bar(x + (j - 1) * w, [RANGE[k][j] for k in knobs], w, label=p, color=cols[p])
ax.set_xticks(x); ax.set_xticklabels(knobs, rotation=20); ax.set_ylabel('F0 range over sweep (x1000)')
ax.set_title('Knob sensitivity (F0 max-min over each sweep, 10 ev). U (induction) is the sensitive '
             'plane; kgate/detector/dilate/baseline/n_iter matter; khi/minc/reducer ~irrelevant.',
             fontsize=10, fontweight='bold')
ax.legend(); ax.grid(alpha=0.2, axis='y')
fig.tight_layout(); pth = os.path.join(cc.FIGDIR, 'fig26_knob_sensitivity.png')
fig.savefig(pth, dpi=130); plt.close(fig); print('saved', pth)
