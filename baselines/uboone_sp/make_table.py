#!/usr/bin/env python3
"""ladder.json -> markdown tables (results/ladder.md), for the README."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "ladder.json")))
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results", "ladder.md")

DESC = {"R0syn": "hits (*) fitted R, float (no ZS) + oracle ROI",
        "R0zs": "R0syn rounded + \\|v\\|<2 dropped + oracle ROI",
        "R0": "stored clean sensor + oracle ROI",
        "R0ex": "R0 with the example's kernel",
        "R1": "+ incoherent noise (coherent removed perfectly)",
        "R2": "+ coherent noise, our gate removal",
        "R2mp": "R2 with multipass removal",
        "R3": "R2 input + MicroBooNE ROI finder (full chain)",
        "R3inc": "R1 input + MicroBooNE ROI finder",
        "R0synnr": "R0syn, no ROI mask",
        "R0zsnr": "R0zs, no ROI mask",
        "R0nr": "R0, no ROI mask (noise-free: artefacts only)",
        "R1nr": "R1, no ROI mask (Gaussian output everywhere)",
        "R2nr": "R2, no ROI mask",
        "R0nrp": "R0nr, readout zero-padded before the FFT",
        "R1nrp": "R1nr, readout zero-padded before the FFT",
        "R2nrp": "R2nr, readout zero-padded before the FFT",
        "model_cpu": "our model, CPU replay of eval_denoise",
        "model_dnjsonl": "our model, dn.jsonl (ab_ft1024_win)"}
names = [r for r in ("R0syn", "R0zs", "R0", "R0ex", "R1", "R2", "R2mp", "R3", "R3inc", "R0synnr", "R0zsnr", "R0nr", "R1nr", "R2nr", "R0nrp", "R1nrp", "R2nrp", "model_cpu",
                     "model_dnjsonl") if r in res]


def v(r, k, ci=False):
    x = res[r]["metrics_covered"].get(k)
    if x is None:
        return "-"
    if ci and x[1] == x[1]:
        return f"{x[0]:.3f} ({x[1]:.3f}-{x[2]:.3f})"
    return f"{x[0]:.3f}" if x[0] == x[0] else "nan"


L = []
L.append(f"Events: {res['_meta']['n_events']} (truth_v2 ev{min(res['_meta']['events'])}-ev{max(res['_meta']['events'])}), "
         f"{res['_meta']['n_windows']} windows, {res['_meta']['n_rows_covered']} covered map rows, Q0 = {res['_meta']['Q0']:.1f} e-.\n")
L.append("### Headline (eval_denoise keys; 16-84% event bootstrap in brackets)\n")
L.append("| rung | what | map r | eff@1% <0.1 | eff@1% 0.1-0.2 | eff@1% 0.2-0.5 | AUC <0.1 | AUC 0.1-0.2 |")
L.append("|---|---|---|---|---|---|---|---|")
for r in names:
    L.append(f"| {r} | {DESC[r]} | {v(r, 'map_r', True)} | {v(r, 'floor_eff1pct_0-0.1', True)} | "
             f"{v(r, 'floor_eff1pct_0.1-0.2', True)} | {v(r, 'floor_eff1pct_0.2-0.5', True)} | "
             f"{v(r, 'floor_auc_0-0.1')} | {v(r, 'floor_auc_0.1-0.2')} |")
L.append("\n### Secondary eval_denoise keys\n")
L.append("| rung | eff@5% <0.1 | eff@5% 0.1-0.2 | eff@5% 0.2-0.5 | AUC 0.2-0.5 | presence AUC | faint AUC | loc w/t 0.1-0.2 | loc w/t 0.2-0.5 | loc w/t >1 |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for r in names:
    L.append(f"| {r} | {v(r, 'floor_eff5pct_0-0.1')} | {v(r, 'floor_eff5pct_0.1-0.2')} | {v(r, 'floor_eff5pct_0.2-0.5')} | "
             f"{v(r, 'floor_auc_0.2-0.5')} | {v(r, 'presence_auc')} | {v(r, 'faint_auc')} | "
             f"{v(r, 'loc_w_0.1-0.2')} / {v(r, 'loc_t_0.1-0.2')} | {v(r, 'loc_w_0.2-0.5')} / {v(r, 'loc_t_0.2-0.5')} | "
             f"{v(r, 'loc_w_1-1e+09')} / {v(r, 'loc_t_1-1e+09')} |")
L.append("\n### Floor per plane type (diagnostic: each plane's OWN 1%-FPR background threshold; not an eval_denoise key)\n")
L.append("| rung | U eff@1% <0.1 / 0.1-0.2 / 0.2-0.5 | V | Y | bg score q50 / q99 (e-) U, V, Y |")
L.append("|---|---|---|---|---|")
for r in names:
    pp = res[r]["metrics_covered"].get("per_plane")
    if not pp:
        continue
    f = lambda p: " / ".join(f"{pp[p][f'floor_eff1pct_{b}']:.2f}" for b in ("0-0.1", "0.1-0.2", "0.2-0.5"))
    q = ", ".join(f"{pp[p]['bg_q50_q99'][0]:.0f}/{pp[p]['bg_q50_q99'][1]:.0f}" for p in "UVY")
    L.append(f"| {r} | {f('U')} | {f('V')} | {f('Y')} | {q} |")
L.append("\n### Per-cell charge resolution (covered map rows with truth charge; median pred/true, 68% half-width of (pred-true)/true, fraction predicted 0)\n")
bins = None
for r in names:
    c = res[r].get("cellres_covered")
    if c:
        bins = list(c)
        break
L.append("| rung | " + " | ".join(f"q {b}" for b in bins) + " |")
L.append("|---|" + "---|" * len(bins))
for r in names:
    c = res[r].get("cellres_covered")
    if not c:
        continue
    L.append(f"| {r} | " + " | ".join(f"{c[b]['median_ratio']:.2f} ± {c[b]['hw68']:.2f} (0: {c[b]['zero_frac']:.2f})" if b in c else "-"
                                       for b in bins) + " |")
L.append("\n### Charge recovery per plane type (all 128 events; classical rungs only)\n")
L.append("| rung | sum charge / sum hits U, V, Y | sum clipped cells / hits U, V, Y | F0 vs filtered truth U, V, Y | per-wire (truth > 20k e-) median / 68% hw U, V, Y | ROI pixel fraction U, V, Y |")
L.append("|---|---|---|---|---|---|")
for r in names:
    P = res[r].get("planes")
    if not P:
        continue
    g = lambda k: ", ".join(f"{P[p][k]:.3f}" for p in "UVY")
    w = ", ".join(f"{P[p]['wire_q>20000']['median']:+.3f} / {P[p]['wire_q>20000']['hw68']:.3f}" if "wire_q>20000" in P[p] else "-"
                  for p in "UVY")
    L.append(f"| {r} | {g('charge_over_hits')} | {g('cells_over_hits')} | {g('F0_f')} | {w} | {g('roi_frac')} |")
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
