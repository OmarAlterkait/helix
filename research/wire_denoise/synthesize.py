"""Collect all method R-D points and produce per-plane frontier plots + summary.

Reads artifacts/*.json, plots compression-vs-F0 and compression-vs-noise_rms per
plane, prints a comparison table at matched compression (~10x). Stage tag in name.
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STAGE = os.environ.get('STAGE', 'A')   # A=intrinsic, B=+coherent
SUFFIX = '' if STAGE == 'A' else '_B'


def load(name):
    p = f'artifacts/{name}{SUFFIX}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def points_of(blob, pt):
    if blob is None or pt not in blob.get('results', blob):
        d = blob.get('results', blob) if blob else {}
        if pt not in d:
            return []
    d = blob.get('results', blob)
    return d[pt].get('points', []) if pt in d else []


def frontier(points, xkey='compression', ykey='f0', maximize=True):
    """Upper-left frontier: for each method, sort by compression."""
    pts = sorted(points, key=lambda p: p[xkey])
    return [p[xkey] for p in pts], [p[ykey] for p in pts]


def main():
    base = load('baselines'); klt = load('klt'); dct = load('dict'); lw = load('lwave')
    methods = {'DWT (best fam)': None, 'KLT': klt, 'dict': dct, 'learned-wavelet': lw}
    summary = {}
    for pt in ['Y', 'U', 'V']:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        # DWT: pick best wavelet family per level from baselines, plot frontier (best F0 per comp bin)
        if base:
            dwt = [p for p in base[pt]['points'] if p['method'].startswith('dwt')]
            # frontier: best F0 at each compression (envelope)
            dwt_sorted = sorted(dwt, key=lambda p: p['compression'])
            ax[0].scatter([p['compression'] for p in dwt], [p['f0'] for p in dwt], s=8, alpha=0.3, color='gray', label='DWT (all)')
            ax[1].scatter([p['compression'] for p in dwt], [p['noise_rms'] for p in dwt], s=8, alpha=0.3, color='gray')
            tt = [p for p in base[pt]['points'] if p['method'] == 'time-threshold']
            x, y = frontier(tt); ax[0].plot(x, y, 'o--', color='black', ms=4, label='time-threshold')
            ax[0].axhline(base[pt]['raw_f0'], color='red', ls=':', lw=1, label='raw (no denoise)')
        for name, blob in [('KLT', klt), ('dict', dct), ('learned-wavelet', lw)]:
            pts = points_of(blob, pt)
            if not pts:
                continue
            x, y = frontier(pts, ykey='f0'); ax[0].plot(x, y, 'o-', ms=4, label=name)
            x, yn = frontier(pts, ykey='noise_rms'); ax[1].plot(x, yn, 'o-', ms=4, label=name)
        ax[0].set(xlabel='compression x', ylabel='F0', title=f'{pt} plane — fidelity (Stage {STAGE})', xscale='log')
        ax[1].set(xlabel='compression x', ylabel='noise RMS (ADC)', title=f'{pt} plane — residual noise', xscale='log')
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f'figures/rd_{pt}{SUFFIX}.png', dpi=110); plt.close(fig)
        # summary @ ~10x
        def at10(pts):
            cand = [p for p in pts if 8 <= p['compression'] <= 13]
            return max(cand, key=lambda p: p['f0']) if cand else None
        row = {}
        if base:
            row['DWT'] = at10([p for p in base[pt]['points'] if p['method'].startswith('dwt')])
        for name, blob in [('KLT', klt), ('dict', dct), ('lwave', lw)]:
            row[name] = at10(points_of(blob, pt))
        summary[pt] = row
    # print table
    print(f"\n=== Stage {STAGE}: F0 / noise_rms @ ~10x compression ===")
    print(f"{'plane':5s} " + " ".join(f"{m:>22s}" for m in ['DWT', 'KLT', 'dict', 'lwave']))
    for pt, row in summary.items():
        cells = []
        for m in ['DWT', 'KLT', 'dict', 'lwave']:
            p = row.get(m)
            cells.append(f"{p['compression']:.0f}x F0{p['f0']:.3f} n{p['noise_rms']:.2f}" if p else " " * 22)
        print(f"{pt:5s} " + " ".join(f"{c:>22s}" for c in cells))
    print(f"\nfigures: figures/rd_{{Y,U,V}}{SUFFIX}.png")
    return summary


if __name__ == '__main__':
    main()
