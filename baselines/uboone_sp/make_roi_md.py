#!/usr/bin/env python3
"""results/roi_sweep.json -> results/roi_sweep.md (tables for results/roi.md)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "roi_sweep.json")))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", "roi_sweep.md")
sets = res["_meta"]["settings"]
EB4 = ("0-0.1", "0.1-0.2", "0.2-0.5", "0.5-1")
SNR = ("nosignal", "1e-09-3", "3-5", "5-7", "7-10", "10-15", "15-inf")


def f(x, d=3):
    return "-" if x is None or x != x else f"{x:.{d}f}"


L = [f"{res['_meta']['n_events']} events, {res['_meta']['n_windows']} windows (floor + near-activity); model windows "
     f"identical to ours: {res['_meta']['model_windows_identical']}.\n",
     "### Every setting: ROI operating point and score-thresholded efficiency at 1% FPR\n",
     "`roi_fpr` = fraction of noise windows with ANY ROI charge; `roi_eff` = fraction of isolated-deposit windows with any ROI charge. "
     "`eff@1%` uses the 99th percentile of the far (bg) or near (bgn) window scores as the threshold.\n",
     "| setting | roi_fpr far / near | roi_eff <0.1 / 0.1-0.2 / 0.2-0.5 / 0.5-1 | far eff@1% <0.1 / 0.1-0.2 / 0.2-0.5 | near eff@1% <0.1 / 0.1-0.2 / 0.2-0.5 | near_fpr_at_far1pct |",
     "|---|---|---|---|---|---|"]
for s in sets + ["model"]:
    m = res[s]["metrics"]
    L.append(f"| {s} | {f(m.get('roi_fpr_far'))} / {f(m.get('roi_fpr_near'))} | "
             + " / ".join(f(m.get(f"roi_eff_{e}")) for e in EB4) + " | "
             + " / ".join(f(m.get(f"floor_eff1pct_{e}")) for e in EB4[:3]) + " | "
             + " / ".join(f(m.get(f"near_eff1pct_{e}")) for e in EB4[:3]) + f" | {f(m.get('near_fpr_at_far1pct'))} |")

L += ["\n### Best efficiency at 1% FPR over settings (chosen on these test events: optimistic for the classical chains)\n",
      "| family | far <0.1 | far 0.1-0.2 | far 0.2-0.5 | near <0.1 | near 0.1-0.2 | near 0.2-0.5 |", "|---|---|---|---|---|---|---|"]
fams = {"example chain (ex_s*)": [s for s in sets if s.startswith("ex_s")],
        "Wire-Cell port, all settings (wc_*)": [s for s in sets if s.startswith("wc_") and s != "wc_oracle"]}
for fam, ss in fams.items():
    row = []
    for side in ("floor", "near"):
        for e in EB4[:3]:
            k = f"{side}_eff1pct_{e}"
            best = max(ss, key=lambda s: res[s]["metrics"].get(k, -1))
            row.append(f"{f(res[best]['metrics'].get(k))} ({best})")
    L.append(f"| {fam} | " + " | ".join(row) + " |")
for h in [k for k in res if k.startswith("hyb_")]:
    m = res[h]["metrics"]
    L.append(f"| post-hoc hybrid {h} (U,V: wc_fixU; Y: wc_nofake) | " + " | ".join(f(m.get(f"{side}_eff1pct_{e}")) for side in ("floor", "near") for e in EB4[:3]) + " |")
m = res["model"]["metrics"]
L.append("| model M4n_ft_full_near (presence) | " + " | ".join(f(m.get(f"{side}_eff1pct_{e}")) for side in ("floor", "near") for e in EB4[:3]) + " |")

L += ["\n### Efficiency by deposit SNR (clean isolated deposits) at the far / near 1%-FPR threshold\n",
      "| setting | " + " | ".join(f"SNR {b}" for b in SNR) + " |", "|---|" + "---|" * len(SNR)]
best_near = {fam: max(ss, key=lambda s: res[s]["metrics"].get("near_eff1pct_0.1-0.2", -1)) for fam, ss in fams.items()}
for s in ["ex_s1", "wc_s1", "wc_fixU"] + sorted(set(best_near.values()) - {"ex_s1", "wc_s1"}) + [k for k in res if k.startswith("hyb_")] + ["model"]:
    m = res[s]["metrics"]
    L.append(f"| {s} | " + " | ".join(f"{f(m.get(f'snr_eff1pct_far_{b}'), 2)} / {f(m.get(f'snr_eff1pct_near_{b}'), 2)}" for b in SNR) + " |")

L += ["\n### Per plane (each plane's own thresholds): far eff@1% / near eff@1% at 0.1-0.2 MeV, roi_fpr far / near, roi_eff\n",
      "| setting | U | V | Y |", "|---|---|---|---|"]
for s in [x for x in ["ex_s1", "wc_s1", "wc_fixU", "wc_nofake"] if x in sets] + sorted(set(best_near.values()) - {"ex_s1", "wc_s1"}) + ["model"]:
    m = res[s]["metrics"]
    cells = []
    for p in "UVY":
        q = m.get(f"plane_{p}", {})
        cells.append(f"{f(q.get('floor_eff1pct_0.1-0.2'), 2)} / {f(q.get('near_eff1pct_0.1-0.2'), 2)}; "
                     f"fpr {f(q.get('roi_fpr_far'), 2)} / {f(q.get('roi_fpr_near'), 2)}; eff {f(q.get('roi_eff'), 2)}")
    L.append(f"| {s} | " + " | ".join(cells) + " |")

L += ["\n### ROI-level decomposition per isolated deposit (classical): ROI efficiency (any truth pixel inside a found ROI), "
      "fraction of the deposit's truth charge inside the ROI (if found), reconstructed / true charge in its window box (if found)\n",
      "| setting | plane | <0.1 | 0.1-0.2 | 0.2-0.5 | 0.5-1 |", "|---|---|---|---|---|---|"]
for s in ["ex_s1", "wc_s1", "wc_fixU"] + sorted(set(best_near.values()) - {"ex_s1", "wc_s1"}):
    for p in ("U", "V", "Y", "all"):
        r = res[s]["decomposition"].get(p, {})
        L.append(f"| {s} | {p} | " + " | ".join(
            (f"{f(r[f'E{e}']['roi_eff'], 2)}; in {f(r[f'E{e}']['q_in_roi_frac_if_found'], 2)}; q {f(r[f'E{e}']['reco_over_true_if_found'], 2)} (n={r[f'E{e}']['n']})"
             if f"E{e}" in r else "-") for e in EB4) + " |")
L += ["\n### ROI efficiency vs deposit SNR (clean deposits), per plane\n",
      "| setting | plane | " + " | ".join(SNR) + " |", "|---|---|" + "---|" * len(SNR)]
for s in ["ex_s1", "wc_s1"] + sorted(set(best_near.values()) - {"ex_s1", "wc_s1"}):
    for p in ("U", "V", "Y", "all"):
        r = res[s]["decomposition"].get(p, {})
        L.append(f"| {s} | {p} | " + " | ".join(f(r.get(f"SNR{b}", {}).get("roi_eff"), 2) for b in SNR) + " |")
md = res["model"].get("decomposition", {})
for lab in ("far1pct", "near1pct"):
    if lab not in md:
        continue
    L.append(f"| model (presence > {md[lab]['threshold']:.3f}, its {lab} threshold) | " + " | ".join(
        f"{p}: " + ", ".join(f(md[lab]['rows'].get(p, {}).get(f'SNR{b}', {}).get('roi_eff'), 2) for b in SNR) for p in ("all",)) + " |")
L += ["\n### Charge inside found ROIs vs inside oracle ROIs (deposits the default setting found): reco/true in the window box, median and 68% half-width\n",
      "| chain | plane, energy | n | found ROI | oracle ROI | truth charge inside found ROI (median frac) |", "|---|---|---|---|---|---|"]
for chain in ("ex", "wc"):
    for k, v in res.get(f"_charge_{chain}", {}).items():
        L.append(f"| {chain} | {k} | {v['n']} | {f(v['found_median'], 2)} ± {f(v['found_hw68'], 2)} | {f(v['oracle_median'], 2)} ± {f(v['oracle_hw68'], 2)} | {f(v['truth_in_found_roi'], 2)} |")
L += ["\n### ROI purity (all ROI pixels of all test events)\n",
      "| setting | U: area truth-empty / +charge truth-empty / truth charge in ROI | V | Y |", "|---|---|---|---|"]
for s in ["ex_oracle", "wc_oracle", "ex_s1", "wc_s1", "wc_fixU"] + sorted(set(best_near.values()) - {"ex_s1", "wc_s1"}):
    pu = res[s]["purity"]
    L.append(f"| {s} | " + " | ".join(f"{f(pu[p]['frac_area_truth_empty'], 2)} / {f(pu[p]['frac_pos_charge_truth_empty'], 2)} / {f(pu[p]['frac_truth_charge_in_roi'], 2)}"
                                     for p in "UVY") + " |")
if md and "far1pct" in md:
    L += ["\n### The model's 'ROI' (cells with presence above its own 1%-FPR window threshold)\n",
          "| threshold | plane | ROI eff <0.1 / 0.1-0.2 / 0.2-0.5 / 0.5-1 | truth charge in ROI if found (0.1-0.2) | covered cells truth-empty | window fpr far / near |",
          "|---|---|---|---|---|---|"]
    for lab in ("far1pct", "near1pct"):
        for p in ("U", "V", "Y", "all"):
            r = md[lab]["rows"].get(p, {})
            L.append(f"| {lab} ({md[lab]['threshold']:.3f}) | {p} | " + " / ".join(f(r.get(f"E{e}", {}).get("roi_eff"), 2) for e in EB4) +
                     f" | {f(r.get('E0.1-0.2', {}).get('q_in_roi_frac_if_found'), 2)} | {f(r.get('purity', {}).get('frac_cells_truth_empty'), 2)} | "
                     f"{f(md[lab]['window_fpr_far'], 3)} / {f(md[lab]['window_fpr_near'], 3)} |")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
