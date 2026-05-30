"""Aggregate + Pareto-analyze a sweep stage. Writes depth_wavelets.json for the
next stage and prints frontier / best configs.

Usage: python scripts/optical/analyze_big.py <stage>
"""
import sys, json, glob, os
import numpy as np

OUT = "/sdf/group/neutrino/omara/helix/temp/figures/big"
stage = sys.argv[1] if len(sys.argv) > 1 else "breadth"
DIST = sys.argv[2] if len(sys.argv) > 2 else "rms_vs_ref"   # distortion metric

rows = [json.loads(l) for f in sorted(glob.glob(f"{OUT}/{stage}_gpu*.jsonl")) for l in open(f)]
ok = [r for r in rows if "error" not in r]
err = [r for r in rows if "error" in r]
print(f"{stage}: {len(rows)} rows, {len(ok)} ok, {len(err)} errors")
if err:
    from collections import Counter
    c = Counter((r.get("wavelet"), r.get("level")) for r in err)
    print("  error groups:", dict(list(c.items())[:6]))


def key(r):
    return (r["wavelet"], r["level"], r["method"], r.get("func", ""), r.get("kappa", ""),
            r.get("e", ""), r.get("keep", ""), r["bits"])


agg = {}
for r in ok:
    agg.setdefault(key(r), []).append(r)
A = []
for k, v in agg.items():
    A.append(dict(wavelet=k[0], level=k[1], method=k[2], func=k[3], kappa=k[4], bits=k[7],
                  n=len(v),
                  comp=float(np.mean([x["comp"] for x in v])),
                  peak=float(np.mean([x["peak_err"] for x in v])),
                  rms=float(np.mean([x["rms_vs_ref"] for x in v])),
                  rel=float(np.mean([x["rel_vs_noisy"] for x in v]))))


def pareto(items, dkey):
    """max comp at each distortion (lower-left frontier of (dist, comp))."""
    pts = sorted(items, key=lambda r: -r["comp"])
    best = 1e9; fr = []
    for r in pts:
        if r[dkey] < best:
            fr.append(r); best = r[dkey]
    return sorted(fr, key=lambda r: r["comp"])


print(f"\n=== global Pareto frontier (comp vs {DIST}) ===")
fr = pareto(A, "rms" if DIST == "rms_vs_ref" else "peak")
print(f"{'comp':>7s} {DIST:>10s} {'peak':>8s} {'wavelet':>9s} {'L':>3s} {'method':>9s} {'func':>8s} {'k':>4s} {'B':>3s}")
for r in fr:
    print(f"{r['comp']:>6.1f}x {(r['rms'] if DIST=='rms_vs_ref' else r['peak']):>10.4f} {r['peak']:>8.4f} "
          f"{r['wavelet']:>9s} {r['level']:>3d} {r['method']:>9s} {str(r['func']):>8s} {str(r['kappa']):>4s} {r['bits']:>3d}")

# best wavelets: at a fixed sensible operating band (rms<0.02), which wavelets give most comp
band = [r for r in A if r["rms"] < 0.02 and r["peak"] < 0.005]
byw = {}
for r in band:
    byw.setdefault(r["wavelet"], []).append(r["comp"])
print(f"\n=== wavelets ranked by max compression at rms<0.02 & peak<0.5% ===")
ranked = sorted(byw.items(), key=lambda kv: -max(kv[1]))
for w, cs in ranked:
    print(f"  {w:>9s}: max {max(cs):.1f}x ({len(cs)} configs in band)")
top_w = [w for w, _ in ranked[:4]]
if top_w:
    json.dump(top_w, open(f"{OUT}/depth_wavelets.json", "w"))
    print(f"\n-> wrote depth_wavelets.json: {top_w}")

# export curated frontier configs for the entropy-coded final stage
if stage == "depth":
    frP = pareto(A, "peak")
    frR = pareto(A, "rms")
    seen, final = set(), []
    for r in sorted(frP + frR, key=lambda r: r["comp"]):
        k = (r["wavelet"], r["level"], r["method"], r["func"], r["kappa"], r["bits"])
        if k in seen:
            continue
        seen.add(k)
        final.append(dict(method=r["method"], func=r["func"], kappa=r["kappa"],
                          wavelet=r["wavelet"], level=r["level"], bits=r["bits"]))
    json.dump(final, open(f"{OUT}/final_configs.json", "w"))
    print(f"\n-> wrote final_configs.json: {len(final)} frontier configs for entropy stage")

# method comparison at the band
bym = {}
for r in band:
    bym.setdefault((r["method"], r["func"]), []).append(r["comp"])
print(f"\n=== methods ranked (max comp at rms<0.02 & peak<0.5%) ===")
for (m, f), cs in sorted(bym.items(), key=lambda kv: -max(kv[1])):
    print(f"  {m:>9s}/{str(f):8s}: max {max(cs):.1f}x")
