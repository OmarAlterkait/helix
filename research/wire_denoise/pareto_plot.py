"""Pareto frontier plots from pareto.json (reads consolidated results, no recompute).
Per plane: F0 vs compression (all combos faint, colored by family), Pareto front,
leanest@best-F0 and RECOMMENDED (deeper-level) markers; second panel = bias."""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

STAGE = os.environ.get('STAGE', 'A').upper()
JSON_IN = 'artifacts/pareto.json' if STAGE == 'A' else f'artifacts/pareto_{STAGE}.json'
SUF = '' if STAGE == 'A' else f'_{STAGE}'
TITLE = 'Stage A (intrinsic)' if STAGE == 'A' else 'Stage B (coherent + helix removal)'
R = json.load(open(JSON_IN))


def family(w):
    for f in ['haar', 'dmey', 'bior', 'rbio', 'coif', 'sym', 'db']:
        if w.startswith(f):
            return f
    return w


FAMS = ['db', 'sym', 'coif', 'bior', 'rbio', 'haar', 'dmey']
col = {f: plt.cm.tab10(i / 10) for i, f in enumerate(FAMS)}

for pt in ['Y', 'U', 'V']:
    if pt not in R:
        continue
    pts = R[pt]['points']; fr = R[pt]['pareto']; sel = R[pt]['select']
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    for f in FAMS:
        wp = [p for p in pts if family(p['w']) == f]
        if wp:
            ax[0].scatter([p['compression'] for p in wp], [p['f0'] for p in wp],
                          s=9, alpha=0.30, color=col[f], label=f)
    ax[0].plot([p['compression'] for p in fr], [p['f0'] for p in fr],
               'k-', lw=1.8, label='Pareto front', zorder=4)
    for tag, mk, c in [('leanest', 'D', 'magenta'), ('recommended', '*', 'lime')]:
        s = sel[tag]
        ax[0].scatter([s['compression']], [s['f0']], marker=mk, s=240 if mk == '*' else 90,
                      edgecolor='k', color=c, zorder=6,
                      label=f"{tag}: {s['w']} L{s['lv']} k{s['k']:g} ({s['compression']:.0f}x)")
    ax[0].set(xscale='log', xlabel='compression x (= pixels / #coeffs)', ylabel='F0',
              title=f'{pt} plane - Pareto: F0 vs #coefficients ({TITLE})')
    ax[0].legend(fontsize=7, ncol=2, loc='lower left'); ax[0].grid(alpha=0.3)
    ax[1].scatter([p['compression'] for p in pts], [p['bias'] for p in pts], s=8, alpha=0.2, color='gray')
    ax[1].plot([p['compression'] for p in fr], [p['bias'] for p in fr], 'k-o', lw=1.5, ms=3, label='front bias')
    ax[1].axhspan(-0.3, 0.3, color='g', alpha=0.08); ax[1].axhline(0, color='g', lw=1)
    ax[1].set(xscale='log', xlabel='compression x', ylabel='bias (recon-clean on signal, ADC)',
              title=f'{pt} - bias vs compression'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f'figures/pareto_{pt}{SUF}.png', dpi=110); plt.close(fig)
    print(f'saved figures/pareto_{pt}{SUF}.png')
