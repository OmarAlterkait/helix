#!/usr/bin/env python3
"""Markdown tables from v2/results/{benchmark,aspects}.json (printed; pasted into
results/validity.md and results/aspects.md)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def f3(b, k):
    v = b.get(k)
    return "-" if v is None else f"{v[0]:.3f} [{v[1]:.3f}-{v[2]:.3f}]"


def s2(b, k):
    v = b.get(k)
    return "-" if v is None else f"{v[0]:.2f}"


def bench():
    B = json.load(open(os.path.join(RES, "benchmark.json")))
    T = B["test"]
    print("## picks\n", json.dumps(B["selection"]["pick"]))
    print("\n## headline (test, 128 events)\n")
    print("| method | near eff@1% 0.1-0.2 MeV, per-plane thr | far, per-plane thr | near, ONE pooled thr (eval) | far, pooled (eval) | near <0.1 / 0.2-0.5 (pp) | near_fpr_at_far1pct (pp) |")
    print("|---|---|---|---|---|---|---|")
    for n in T:
        b = T[n]["boot"]
        print(f"| {n} | {f3(b, 'near_pp_0.1-0.2')} | {f3(b, 'far_pp_0.1-0.2')} | {f3(b, 'near_pool_0.1-0.2')} | {f3(b, 'far_pool_0.1-0.2')} | "
              f"{s2(b, 'near_pp_0-0.1')} / {s2(b, 'near_pp_0.2-0.5')} | {s2(b, 'near_fpr_at_far1pct_pp')} |")
    print("\n## per plane, 0.1-0.2 MeV, near / far (own thresholds)\n")
    print("| method | U | V | Y |\n|---|---|---|---|")
    for n in T:
        b = T[n]["boot"]
        print(f"| {n} | " + " | ".join(f"{f3(b, f'near_{p}_0.1-0.2')} / {s2(b, f'far_{p}_0.1-0.2')}" for p in "UVY") + " |")
    print("\n## per plane near, all energy bins\n")
    print("| method | plane | <0.1 | 0.1-0.2 | 0.2-0.5 | 0.5-1 |\n|---|---|---|---|---|---|")
    for n in T:
        b = T[n]["boot"]
        for p in "UVY":
            print(f"| {n} | {p} | " + " | ".join(s2(b, f"near_{p}_{e}") for e in ("0-0.1", "0.1-0.2", "0.2-0.5", "0.5-1")) + " |")
    print("\n## SNR (pp), near / far\n")
    sn = ("nosignal", "0-3", "3-5", "5-7", "7-10", "10-15", "15-inf")
    print("| method | " + " | ".join(sn) + " |\n|" + "---|" * (len(sn) + 1))
    for n in T:
        b = T[n]["boot"]
        print(f"| {n} | " + " | ".join(f"{s2(b, f'near_pp_snr_{s}')} / {s2(b, f'far_pp_snr_{s}')}" for s in sn) + " |")
    print("\n## eval-standard keys (std)\n")
    for n in T:
        s = T[n].get("std")
        if s:
            ks = ("floor_eff1pct_0.1-0.2", "near_eff1pct_0.1-0.2", "near_eff1pct_0-0.1", "near_eff1pct_0.2-0.5", "near_fpr_at_far1pct",
                  "loc_w_0.2-0.5", "loc_t_0.2-0.5", "loc_w_0.5-1", "loc_t_0.5-1")
            print(n, {k: s.get(k, [None])[0] for k in ks})
    print("\n## centroid offsets (pred - truth), iso >= 0.2 MeV\n")
    for n in T:
        o = T[n].get("off", {})
        print(n, {p: (round(v["dw_mean"], 3), round(v["dw_sem"], 3), round(v["dt_mean"], 2), round(v["dt_sem"], 2)) for p, v in o.items()})
    print("\n## tuning-set candidates (top 6 per plane)\n")
    for p, rows in B["selection"]["candidates"].items():
        print(p, [(r["stat"], round(r["far"], 3), round(r["near"], 3)) for r in rows[:6]])
    print("\nmodel repro:", B["model_repro"])


def aspects():
    A = json.load(open(os.path.join(RES, "aspects.json")))
    print("\n# ASPECTS\n")
    print("coverage", A["coverage"])
    names = ("cls_default", "cls_tuned_q", "cls_noroi", "cls_oracle_roi", "M4n_q", "N64_q")
    print("\n## empty cells: q99 (e-) / frac > 0.25 Q0 / frac > 1 Q0, by distance (cells)\n")
    for p in "UVY":
        print(f"\n{p}\n| method | " + " | ".join(("1", "2", "3-4", "5-8", "9-16", ">16", "all")) + " |\n|" + "---|" * 8)
        for n in names:
            row = []
            for d in ("1", "2", "3-4", "5-8", "9-16", ">16", "all"):
                r = A["noise"].get(f"{n}|{p}|{d}")
                row.append("-" if not r else f"{r['q50_90_99_999'][2]:.0f} / {r['frac_gt_025Q0']:.3f} / {r['frac_gt_1Q0']:.4f}")
            print(f"| {n} | " + " | ".join(row) + " |")
    print("\n## empty-cell counts > 0.05 / 1 Q0 (all distances)\n")
    for n in names:
        print(n, {p: (A["noise"][f"{n}|{p}|all"]["count_gt_005Q0"], A["noise"][f"{n}|{p}|all"]["count_gt_1Q0"], A["noise"][f"{n}|{p}|all"]["n"]) for p in "UVY"})
    print("\n## fidelity: median ratio / hw68 / frac zero by true charge\n")
    for p in "UVYa":
        bins = list(A["fidelity"][f"M4n_q|{p}"].keys())
        print(f"\n{p}\n| method | " + " | ".join(bins) + " |\n|" + "---|" * (len(bins) + 1))
        for n in names:
            r = A["fidelity"].get(f"{n}|{p}", {})
            print(f"| {n} | " + " | ".join("-" if b not in r else f"{r[b]['median_ratio']:.2f} / {r[b]['hw68']:.2f} / {r[b]['frac_zero']:.2f}" for b in bins) + " |")
    print("\n## totals (all cells / truth cells)\n")
    for n in names:
        print(n, {p: (round(A["totals"][f"{n}|{p}"]["all_cells_over_truth"], 3), round(A["totals"][f"{n}|{p}"]["truth_cells_over_truth"], 3)) for p in "UVY"})
    print("\n## near-activity noise windows above the FAR threshold by distance\n")
    for n in ("cls_default", "cls_tuned_q", "cls_tuned_any", "cls_noroi", "M4n_q", "M4n_p", "N64_q", "N64_p"):
        print(n, {p: [round(A["neardist"].get(f"{n}|{p}|{d}", {}).get("fpr_at_far1pct", float('nan')), 3) for d in ("1-2", "3-4", "5-8", "9-16")] for p in "UVY"})
    print("\n## pair dips (all planes): median dip / frac resolved (dip<0.5) ; truth median dip\n")
    sb = ("w<=4", "w<=8", "w<=16", "w<=24", "t<=32", "t<=64", "t<=96")
    for n in ("cls_default", "cls_tuned_q", "cls_tuned_any", "cls_noroi", "M4n_q", "M4n_p", "N64_q"):
        print(n, [(s, round(A["dip"].get(f"{n}|a|{s}", {}).get("median_dip", float('nan')), 2),
                   round(A["dip"].get(f"{n}|a|{s}", {}).get("frac_resolved", float('nan')), 2)) for s in sb])
    print("truth", [(s, round(A["dip"].get(f"M4n_q|a|{s}", {}).get("truth_median_dip", float('nan')), 2), A["dip"].get(f"M4n_q|a|{s}", {}).get("n")) for s in sb])


if __name__ == "__main__":
    if "aspects" not in sys.argv:
        bench()
    if "bench" not in sys.argv:
        aspects()
