"""Consolidate the 4-metric OAT knob sweep (knob_{Y,U,V}.json) into a single ranked picture:
per-knob range of each metric (F0, kept, nz_in, nz_out) per plane, an aggregate importance
score, and a 4-panel figure. 'most effect / least effect' ranking."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cc_common as cc

PLANES = ['Y', 'U', 'V']
EXCLUDE = {'klo': {0.5}}        # degenerate (percolating) operating points to drop from range
METRICS = ['F0', 'kept', 'nz_in', 'nz_out']
D = {p: json.load(open(f'knob_{p}.json')) for p in PLANES}
KNOBS = list(D['Y']['knobs'])


def vals_for(plane, knob):
    """Return arrays (values, F0, kept, nz_in, nz_out) with degenerate points excluded."""
    rows = [r for r in D[plane]['knobs'][knob] if r[0] not in EXCLUDE.get(knob, set())]
    arr = np.array([[r[1], r[2], r[3], r[4]] for r in rows])  # F0,kept,nz_in,nz_out
    return [r[0] for r in rows], arr


def rng(plane, knob, mi):
    _, a = vals_for(plane, knob)
    return float(a[:, mi].max() - a[:, mi].min())


# ---- per-metric range tables + aggregate fractional importance score ----
defm = {p: D[p]['def'] for p in PLANES}  # [F0,kept,nz_in,nz_out]
# denominators for fractional impact: F0 -> (1-F0) error budget; others -> def value
def frac(plane, knob, mi):
    r = rng(plane, knob, mi)
    base = (1 - defm[plane][0]) if mi == 0 else defm[plane][mi]
    return r / max(base, 1e-9)

score = {}  # knob -> aggregate (mean fractional range over 4 metrics x 3 planes)
for k in KNOBS:
    score[k] = float(np.mean([frac(p, k, mi) for p in PLANES for mi in range(4)]))
order = sorted(KNOBS, key=lambda k: -score[k])

print("\n=========== CONSOLIDATED KNOB SENSITIVITY (range over sweep, 12 ev) ===========")
print("metric ranges per plane:  F0 (x1000) | kept (count) | nz_in (ADC) | nz_out (ADC)\n")
hdr = f"{'knob':>9} | " + " | ".join(f"{m:>21}" for m in METRICS)
print(hdr); print('-' * len(hdr))
for k in order:
    cells = []
    for mi, m in enumerate(METRICS):
        scale = 1000 if m == 'F0' else 1
        trip = "/".join(f"{rng(p, k, mi) * scale:.{0 if m in ('kept',) else (1 if m=='F0' else 2)}f}" for p in PLANES)
        cells.append(f"{trip:>21}")
    print(f"{k:>9} | " + " | ".join(cells) + f"   score={score[k]:.3f}")
print("\n(triplets are Y/U/V; F0 in milli-F0, kept in coeff count, nz in ADC)")
print(f"\nDEF baseline per plane (F0,kept,nz_in,nz_out):")
for p in PLANES:
    print(f"   {p}: {defm[p][0]:.4f} {defm[p][1]:.0f} {defm[p][2]:.3f} {defm[p][3]:.3f}")
print(f"\nRANKING by aggregate fractional impact (most->least): {order}")

# ---- figure: 4 panels, one per metric, grouped bars by knob (sorted by score), colored by plane ----
cols = {'Y': 'tab:orange', 'U': 'tab:green', 'V': 'tab:blue'}
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
units = {'F0': 'F0 range (x1000)', 'kept': 'kept-count range', 'nz_in': 'nz_in range (ADC)', 'nz_out': 'nz_out range (ADC)'}
for mi, (ax, m) in enumerate(zip(axes.ravel(), METRICS)):
    x = np.arange(len(order)); w = 0.27
    for j, p in enumerate(PLANES):
        scale = 1000 if m == 'F0' else 1
        ax.bar(x + (j - 1) * w, [rng(p, k, mi) * scale for k in order], w, label=p, color=cols[p])
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(units[m]); ax.set_title(f"{m}  (knobs sorted by overall impact ->)", fontsize=10, fontweight='bold')
    ax.grid(alpha=0.2, axis='y'); ax.legend(fontsize=8)
fig.suptitle("Consolidated knob sensitivity: range each knob induces in each metric, per plane (12 ev).\n"
             "Left knobs matter most; right knobs are inert. U (induction) is the sensitive plane throughout.",
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
pth = f"{cc.FIGDIR}/fig28_knob_consolidated.png"
fig.savefig(pth, dpi=130); plt.close(fig); print('\nsaved', pth)
