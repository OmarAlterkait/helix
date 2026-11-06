#!/usr/bin/env python3
"""Markdown tables from results/dense_<tag>.json for the v3 models (M6d raw and gated, M6dc gated; M4n reference).

    python baselines/uboone_sp/dense/tables3.py --tag v3 > baselines/uboone_sp/dense/results/tables_v3.md
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DL = ["iso", "<=10k", "10k-100k", "100k-1M", "1M-4M", ">4M"]
QL = ["<500", "0.5-1k", "1-2k", "2-4k", "4-8k", "8-16k", "16-64k", ">64k"]
RQ = ["<1k", "1-4k", "4-16k", "16-64k", "64-256k", "256k-1M", "1-4M", ">4M"]
CH = ["cls_default", "cls_tuned_q", "cls_oracle", "M4n_q", "M4n_g", "M6d_q", "M6d_g", "M6dc_g"]
NM = {"cls_default": "classical default", "cls_tuned_q": "classical tuned", "cls_tuned_any": "classical tuned best stat",
      "cls_oracle": "classical oracle ROI (ref)", "M4n_q": "M4n charge", "M4n_p": "M4n presence", "N64_q": "64-label charge",
      "N64_p": "64-label presence", "M2n_q": "M2n charge", "M2n_p": "M2n presence",
      "M4n_g": "M4n gated", "M6d_q": "M6d raw", "M6d_g": "M6d gated", "M6d_p": "M6d presence", "M6dc_g": "M6dc gated",
      "M6dc_p": "M6dc presence"}


def t(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fit")
    a = ap.parse_args()
    R = json.load(open(os.path.join(HERE, "results", f"dense_{a.tag}.json")))
    F = R["fidelity"]
    print(f"# Tables ({a.tag} kernel), 128 test events, event bootstrap 16-84%\n")
    print("## Counts of charged cells (all planes), density x charge\n")
    print(t(["density \\ charge"] + QL, [[d] + [F["counts"][f"all|{d}|{q}"] for q in QL] for d in DL]))
    for pl in ("all", "U", "V", "Y"):
        print(f"\n## 1. Fidelity, plane {pl}: median pred/true (68% half-width), [fraction zero]\n")
        rows = []
        for m in CH:
            for d in DL[1:]:
                r = [NM[m], d]
                for q in QL:
                    e = F[m].get(f"{pl}|{d}|{q}")
                    r.append(f"{e['median_ratio'][0]:.2f} ({e['hw68'][0]:.2f}) [{e['frac_zero'][0]:.2f}]" if e else "-")
                rows.append(r)
        print(t(["method", "density"] + QL, rows))
    print("\n## 1b. Median |pred-true|/true, all planes, [16-84%]\n")
    rows = []
    for m in CH:
        for d in DL[1:]:
            r = [NM[m], d]
            for q in QL:
                e = F[m].get(f"all|{d}|{q}")
                r.append(f"{e['median_rel_abs_err'][0]:.2f} [{e['median_rel_abs_err'][1]:.2f}-{e['median_rel_abs_err'][2]:.2f}]" if e else "-")
            rows.append(r)
    print(t(["method", "density"] + QL, rows))
    Z = R.get("zeroed", {})
    if Z:
        print("\n## Share of true charge in cells the output sets to 0 (cells share), by density\n")
        print(t(["method", "plane"] + DL + ["all"], [[NM[m], pl] + [f"{Z[f'{m}|{pl}|{d}']['charge_frac']:.3f} ({Z[f'{m}|{pl}|{d}']['cell_frac']:.2f})" for d in DL + ["all"]]
                                                  for m in ("cls_tuned_q", "M4n_q", "M4n_g", "M6d_q", "M6d_g", "M6dc_g") for pl in "UVY"]))
        print("\n## Share of true charge zeroed, by cell charge (all planes, all densities / >1M e- density), [charge zeroed in units of 1e6 e-]\n")
        print(t(["method"] + QL + ["total zeroed / total"], [[NM[m]] + [(f"{Z[k1]['charge_frac']:.3f} / {Z[k2]['charge_frac']:.3f} [{Z[k1]['charge_zeroed'] / 1e6:.0f}]" if (k1 := f"{m}|all|all|{q}") in Z and (k2 := f"{m}|all|1M-4M|{q}") in Z else "-") for q in QL]
                                                     + [f"{Z[f'{m}|all|total_charge_zeroed'] / Z[f'{m}|all|total_charge']:.4f}"] for m in CH]))
    print("\n## 2. Empty cells: fraction above 0.05 Q0 / 1 Q0 by density (all planes) [16-84% of the 1 Q0 fraction]\n")
    fc = R["false_charge"]
    rows = []
    for pl in ("all", "U", "V", "Y"):
        for m in CH:
            rows.append([pl, NM[m]] + [f"{fc[k]['gt005Q0'][0]:.4f} / {fc[k]['gt1Q0'][0]:.5f} [{fc[k]['gt1Q0'][1]:.5f}-{fc[k]['gt1Q0'][2]:.5f}]"
                                       if (k := f"{m}|{pl}|{d}") in fc else "-" for d in DL])
    print(t(["plane", "method"] + DL, rows))
    print("\nempty cells per density bin (all planes): " + ", ".join(f"{d}: {fc[f'M4n_q|all|{d}']['n_empty']:,}" for d in DL))
    fq = R["false_charge_q"]
    print("\n### Predicted charge on empty cells: 90th / 99th / 99.9th percentile (e-), mean, all planes\n")
    print(t(["method"] + DL, [[NM[m]] + [("{:.0f} / {:.0f} / {:.0f}, {:.0f}".format(*fq[k]["q50_90_99_999"][1:], fq[k]["mean"]) if (k := f"{m}|all|{d}") in fq else "-") for d in DL] for m in CH]))
    P = R["presence"]
    for pl in ("all", "U", "V", "Y"):
        print(f"\n## 3. Cell efficiency at 1% FPR set on empty cells of the same density stratum, plane {pl}"
              f"{' (per-plane thresholds, pooled)' if pl == 'all' else ''} [16-84%]\n")
        rows = []
        for m in ("cls_tuned_any", "cls_tuned_q", "M4n_p", "M4n_g", "M6d_p", "M6d_g", "M6dc_p", "M6dc_g"):
            for d in DL[1:]:
                rows.append([NM[m], d] + [f"{e['eff'][0]:.2f}" + (f" [{e['eff'][1]:.2f}-{e['eff'][2]:.2f}]" if q in ("1-2k", "2-4k", "all") else "")
                                          if (e := P.get(f"{m}|{pl}|{d}|{q}")) else "-" for q in QL + ["all"]])
        print(t(["method", "density"] + QL + ["all charges"], rows))
    print("\n### AUC (charged vs empty cells, same density stratum), per plane, all charges\n")
    rows = []
    for pl in "UVY":
        for m in ("cls_tuned_any", "cls_tuned_q", "cls_oracle", "M4n_p", "M4n_q", "M4n_g", "M6d_p", "M6d_q", "M6d_g", "M6dc_p", "M6dc_g"):
            rows.append([pl, NM[m]] + [f"{P[f'{m}|{pl}|{d}|all']['auc'][0]:.3f}" for d in DL])
    print(t(["plane", "method"] + DL, rows))
    for key, lab in (("regions16x8", "16 x 8 cells = 32 wires x 128 ticks"), ("regions4x4", "4 x 4 cells = 8 wires x 64 ticks")):
        G = R[key]
        for pl in ("all", "U", "V", "Y"):
            print(f"\n## 4. Regions ({lab}), plane {pl}: median pred/true (68% hw) / sum ratio, by true region charge\n")
            rows = [[NM[m]] + [f"{e['median_ratio'][0]:.2f} ({e['hw68'][0]:.2f}) / {e['sum_ratio'][0]:.2f}" if (e := G[m].get(f"{pl}|q|{q}")) else "-" for q in RQ]
                    + [f"{G[m][f'{pl}|fit_loglog_T>4k']['slope']:.3f} / {G[m][f'{pl}|fit_loglog_T>4k']['r_log']:.3f}"] for m in CH]
            print(t(["method"] + RQ + ["log-log slope / r (T > 4k)"], rows))
        print(f"\nRegion counts (all planes): " + ", ".join(f"{q}: {G['M4n_q'][f'all|q|{q}']['n']:,}" for q in RQ if f"all|q|{q}" in G["M4n_q"]))
        print(f"\n### {lab}: by the mean density of the region's charged cells, all planes, all region charges: median (hw68) / sum ratio\n")
        print(t(["method"] + DL, [[NM[m]] + [f"{e['median_ratio'][0]:.2f} ({e['hw68'][0]:.2f}) / {e['sum_ratio'][0]:.3f} [{e['sum_ratio'][1]:.3f}-{e['sum_ratio'][2]:.3f}]"
                                              if (e := G[m].get(f"all|d|{d}")) else "-" for d in DL] for m in CH]))
        print(f"\n### {lab}: empty regions (true charge 0), all planes\n")
        print(t(["method", "n", "> 0.25 Q0", "> 1 Q0", "> 4 Q0", "mean predicted (e-)"],
                [[NM[m], f"{(e := G['empty'][f'{m}|all'])['n_empty']:,}", f"{e['frac_gt025Q0']:.4f}", f"{e['frac_gt1Q0']:.4f}", f"{e['frac_gt4Q0']:.5f}", f"{e['mean_pred']:.0f}"] for m in CH]))
    Pp = R["particles"]
    EBL = ["0.1-0.3", "0.3-1", "1-3", "3-10", "10-30", "30-100", ">100"]
    OBL = ["<0.1", "0.1-0.3", "0.3-0.6", ">0.6"]
    print(f"\n## 5. Particles ({Pp['n_particles']:,} particle-plane entries with >= 1 Q0 own charge): pred/true on majority cells, median (68% hw)\n")
    for pl in ("all", "U", "V", "Y"):
        print(f"\nplane {pl}\n")
        print(t(["method"] + [f"E {e} MeV" for e in EBL] + [f"overlap {o}" for o in OBL],
                [[NM[m]] + [f"{e['median_ratio'][0]:.2f} ({e['hw68'][0]:.2f})" if (e := Pp[m].get(f"{pl}|E|{x}")) else "-" for x in EBL]
                 + [f"{e['median_ratio'][0]:.2f} ({e['hw68'][0]:.2f})" if (e := Pp[m].get(f"{pl}|overlap|{x}")) else "-" for x in OBL] for m in CH]))
    print("\ncounts by energy: " + ", ".join(f"{k}: {v:,}" for k, v in Pp["n_by_E"].items()) +
          "; by overlap: " + ", ".join(f"{o}: {Pp['M4n_q'][f'all|overlap|{o}']['n']:,}" for o in OBL))
    print("\n### Energy x overlap, all planes: median (68% hw)\n")
    print(t(["method"] + [f"{e} / {o}" for e in ("1-3", "10-30", ">100") for o in OBL],
            [[NM[m]] + [f"{v['median_ratio'][0]:.2f} ({v['hw68'][0]:.2f})" if (v := Pp[m].get(f"all|EO|{e}|{o}")) else "-" for e in ("1-3", "10-30", ">100") for o in OBL] for m in CH]))
    S = R["separation"]
    for ax in ("wire", "time"):
        print(f"\n## 6. Separation along {ax}: both found / resolved given found / resolved, by density at the valley [resolved 16-84%]\n")
        rows = []
        for pl in ("all", "U", "V", "Y"):
            for m in CH:
                rows.append([pl, NM[m]] + [f"{e['both_found'][0]:.2f} / {e['resolved_given_found'][0]:.2f} / {e['resolved'][0]:.2f}" + (f" [{e['resolved'][1]:.2f}-{e['resolved'][2]:.2f}]" if d == "all" else "")
                                           if (e := S[m].get(f"{pl}|{ax}|sall|{d}")) else "-" for d in DL[1:] + ["all"]])
        print(t(["plane", "method"] + DL[1:] + ["all"], rows))
        print("\npairs: " + ", ".join(f"{d}: {S['counts'].get(f'all|{ax}|sall|{d}', 0):,}" for d in DL[1:] + ["all"]))
        print("\nby separation (all planes, all densities): " + "; ".join(
            f"s={s} cells: " + ", ".join(f"{NM[m]} {S[m][f'all|{ax}|s{s}|all']['resolved'][0]:.2f}" for m in ("cls_tuned_q", "M4n_q", "M4n_g", "M6d_q", "M6d_g", "M6dc_g", "cls_oracle"))
            for s in (2, 3, 4) if f"all|{ax}|s{s}|all" in S["M4n_q"]))
    E = R["event"]
    print("\n## 7. Event level: total predicted / true [16-84%]; per-plane-event ratio median [p16, p84]; median map correlation linear / log / log on the activity footprint\n")
    rows = []
    for pl in ("U", "V", "Y", "all"):
        for m in CH:
            e = E[f"{m}|{pl}"]
            rows.append([pl, NM[m], f"{e['total_ratio'][0]:.3f} [{e['total_ratio'][1]:.3f}-{e['total_ratio'][2]:.3f}]",
                         f"{e['per_plane_ratio_median']:.3f} [{e['per_plane_ratio_p16_p84'][0]:.3f}, {e['per_plane_ratio_p16_p84'][1]:.3f}]",
                         f"{e['corr_lin_median']:.3f} / {e['corr_log_median']:.3f} / {e['corr_log_footprint_median']:.3f}"])
    print(t(["plane", "method", "total ratio", "per plane-event", "r lin / log / log footprint"], rows))
    print("\n## Model token coverage of charged cells, by density (all planes)\n")
    Cv = R["coverage"]
    print(t(["density", "n charged", "cells covered", "charge covered", "empty cells covered"],
            [[d, f"{Cv[f'all|{d}']['n']:,}", f"{Cv[f'all|{d}']['covered']:.3f}", f"{Cv[f'all|{d}']['charge_covered']:.4f}", f"{Cv[f'empty|all|{d}']['covered']:.3f}"] for d in DL]))


if __name__ == "__main__":
    main()
