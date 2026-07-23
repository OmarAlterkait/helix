"""Analyze suppress_sweep.jsonl: per plane, the F0-vs-#coeffs Pareto across ALL
suppression methods/wavelets/levels, the best method at a fixed coeff budget, and
the overall winner (low coeffs, high F0, |bias| small)."""
import json, sys
import numpy as np

pts = [json.loads(l) for l in open('artifacts/suppress_sweep.jsonl') if l.strip()]
BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 60000   # target coeffs/plane
BIASMAX = 0.20


def pareto(P):
    P = sorted([p for p in P if abs(p['bias']) <= BIASMAX], key=lambda p: p['n_kept'])
    fr, best = [], -1
    for p in P:
        if p['f0'] > best + 1e-9:
            fr.append(p); best = p['f0']
    return fr


for pt in ['Y', 'U', 'V']:
    P = [p for p in pts if p['plane'] == pt]
    if not P:
        continue
    print(f"\n========== {pt} plane ({len(P)} combos) ==========")
    # best per method at <= BUDGET coeffs (max F0), bias-limited
    print(f"  best F0 per method at <= {BUDGET:,} coeffs (|bias|<{BIASMAX}):")
    for m in ['uni_hard', 'garrote', 'fixed', 'snr_topk', 'block', 'xscale']:
        cand = [p for p in P if p['method'] == m and p['n_kept'] <= BUDGET and abs(p['bias']) <= BIASMAX]
        if cand:
            b = max(cand, key=lambda p: p['f0'])
            print(f"    {m:9s}: F0={b['f0']:.4f} n_kept={b['n_kept']:>7,} bias={b['bias']:+.3f} "
                  f"[{b['w']} L{b['lv']} p={b['param']}]")
        else:
            print(f"    {m:9s}: (none under budget within bias)")
    # overall Pareto frontier (a few representative points)
    fr = pareto(P)
    print(f"  Pareto frontier (F0 vs coeffs, |bias|<{BIASMAX}):")
    for p in fr:
        if p['n_kept'] <= 4 * BUDGET:
            print(f"    n_kept={p['n_kept']:>7,} comp={p['compression']:6.1f}x F0={p['f0']:.4f} "
                  f"bias={p['bias']:+.3f}  [{p['method']} {p['w']} L{p['lv']} p={p['param']}]")
