#!/usr/bin/env python3
"""One table of the calibrated products (scripts/eval_denoise.py --products, 5-draw aggregate rows)
plus the reference raw rows, from products/results/dn_products_<model>.jsonl -> markdown + json."""
import json
import os
import sys

R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/products/results"
MODELS = ["M4n", "M6d", "M6dc", "lb_dec_hits", "lb_dec_cond_hits", "lb_dec_ct", "lb_dec_cond_ct",
          "sp_dec_c25", "sp_dec_cond_c25", "sp_dec_c102", "sp_dec_cond_c102", "M7d", "M7dc", "M10lin", "M11", "M13"]
P = os.path.join(os.path.dirname(R), "params_{}.json")


def support(m):
    try:
        return float(json.load(open(P.format(m)))[m]["support_min_q"])
    except (OSError, KeyError):
        return None


def rows(m, src):
    f = os.path.join(R, f"dn_products_{m}_c25.jsonl")         # rescore with cal_c25_* (same rows otherwise)
    if not os.path.exists(f):
        f = os.path.join(R, f"dn_products_{m}.jsonl")
    if not os.path.exists(f):
        return None
    out = {}
    for l in open(f):
        r = json.loads(l)
        t = r["tag"]
        if src == "draws" and t.startswith(f"{m}@draws"):
            out[t.split("@draws")[1][1:]] = r["trained"]          # '' / 'pres' / ... (strip K)
        if src == "stored" and "@" not in t:
            out[t[len(m):]] = r["trained"]
    out = {k.lstrip("0123456789"): v for k, v in out.items()}
    if support(m) == 25.0:                                     # own target IS complete q > 25
        for r in out.values():
            for k in [k for k in r if k.startswith("cal_") and not k.startswith(("cal_own_", "cal_c25_"))]:
                r.setdefault("cal_c25_" + k[4:], r[k])
    return out or None


def v(T, row, key, nd=3):
    x = T.get(row, {}).get(key)
    if x is None:
        return "-"
    if isinstance(x, list):
        return f"{x[0]:.{nd}f} [{x[1]:.{nd}f}, {x[2]:.{nd}f}]" if nd and len(x) == 3 else f"{x[0]:.{nd}f}"
    return f"{x:.{nd}f}"


def c(T, row, key, nd=3):
    x = T.get(row, {}).get(key)
    if x is None:
        return "-"
    return f"{(x[0] if isinstance(x, list) else x):.{nd}f}"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "draws"
    lines, J = [], {}
    hdr = ["model", "pres LL raw/cal (complete q>25, shared)", "pres ECE raw/cal (q>25)", "Brier raw/cal (q>25)",
           "pres LL raw/cal (complete q>c)", "pres ECE raw/cal (complete)", "pres LL raw/cal (own label)",
           "pres ECE raw/cal (own)"]
    lines.append("### Calibrated presence (all covered cells)\n")
    lines.append("| " + " | ".join(hdr) + " |"); lines.append("|" + "---|" * len(hdr))
    for m in MODELS:
        T = rows(m, src)
        if T is None:
            continue
        J[m] = T
        lines.append(f"| {m} | {c(T, '_pres', 'cal_c25_logloss', 4)} / {c(T, '_cal', 'cal_c25_logloss', 4)} | "
                     f"{c(T, '_pres', 'cal_c25_ece', 4)} / {c(T, '_cal', 'cal_c25_ece', 4)} | "
                     f"{c(T, '_pres', 'cal_c25_brier', 4)} / {c(T, '_cal', 'cal_c25_brier', 4)} | "
                     f"{c(T, '_pres', 'cal_logloss', 4)} / {c(T, '_cal', 'cal_logloss', 4)} | "
                     f"{c(T, '_pres', 'cal_ece', 4)} / {c(T, '_cal', 'cal_ece', 4)} | "
                     f"{c(T, '_pres', 'cal_own_logloss', 4)} / {c(T, '_cal', 'cal_own_logloss', 4)} | "
                     f"{c(T, '_pres', 'cal_own_ece', 4)} / {c(T, '_cal', 'cal_own_ece', 4)} |")
    lines.append("\n### Calorimetry map: truth / prediction per PREDICTED bin (complete truth), plane totals\n")
    hdr = ["model", "map", "cell 102-500", "cell 1k-2k", "cell 4k-8k", "cell 16k-64k", "region 4k-16k", "region 64k-256k",
           "total U", "total V", "total Y"]
    lines.append("| " + " | ".join(hdr) + " |"); lines.append("|" + "---|" * len(hdr))
    for m, T in J.items():
        for row, nm in (("", "raw"), ("_calo", "calo")):
            lines.append(f"| {m} | {nm} | " + " | ".join(c(T, row, k) for k in (
                "lin_cell_102-500", "lin_cell_1k-2k", "lin_cell_4k-8k", "lin_cell_16k-64k", "lin_region_4k-16k",
                "lin_region_64k-256k", "tot_ratio_U", "tot_ratio_V", "tot_ratio_Y")) + " |")
    lines.append("\n### Clean map vs raw / gated: haze, false and zeroed charge (complete truth, readout)\n")
    hdr = ["model", "map", "haze (empty cells > 0.05 Q0)", "false charge / total", "zeroed charge / total", "L1 / total"]
    lines.append("| " + " | ".join(hdr) + " |"); lines.append("|" + "---|" * len(hdr))
    for m, T in J.items():
        for row, nm in (("", "raw"), ("_gated", "gated"), ("_calo", "calo"), ("_clean", "clean")):
            lines.append(f"| {m} | {nm} | " + " | ".join(c(T, row, k, 4) for k in (
                "fm_haze_frac_0.05Q0", "fm_false_frac", "fm_zeroed_frac", "fm_l1_frac")) + " |")
    lines.append("\n### Detection at 0.1-0.2 MeV, per-plane 1% FPR: isolated positives, near (bgn) and far (bg) thresholds\n")
    hdr = ["model", "score", "isolated near", "isolated far", "all iso near (published def.)"]
    lines.append("| " + " | ".join(hdr) + " |"); lines.append("|" + "---|" * len(hdr))
    for m, T in J.items():
        for row, nm in (("", "raw charge"), ("_pres", "raw presence"), ("_gated", "gated"), ("_cal", "p_cal"),
                        ("_calo", "calo"), ("_clean", "clean")):
            lines.append(f"| {m} | {nm} | {v(T, row, 'near_iso_pp_eff1pct_0.1-0.2')} | {v(T, row, 'floor_iso_pp_eff1pct_0.1-0.2')} | "
                         f"{v(T, row, 'near_pp_eff1pct_0.1-0.2')} |")
    lines.append("\n### Bright cells (complete truth >= 16k e-): median and res68 of pred/truth\n")
    hdr = ["model", "raw med", "raw res68", "calo med", "calo res68", "clean med", "clean res68"]
    lines.append("| " + " | ".join(hdr) + " |"); lines.append("|" + "---|" * len(hdr))
    for m, T in J.items():
        lines.append(f"| {m} | " + " | ".join(c(T, row, k, 4) for row in ("", "_calo", "_clean")
                                              for k in ("ct_med_16k", "ct_res68_16k")) + " |")
    out = "\n".join(lines)
    print(out)
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"products_{src}")
    open(base + ".md", "w").write(out + "\n")
    json.dump(J, open(base + ".json", "w"))


if __name__ == "__main__":
    main()
