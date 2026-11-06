#!/usr/bin/env python3
"""Reduce firstp_sampling.py per-event counts into causes/results/firstp_sampling.json and .md.

    scripts/helix_run.sh python baselines/uboone_sp/causes/firstp_sampling_sum.py \
        --inp /global/cfs/cdirs/m5238/users/oalter/helix_runs/firstp/sampling
"""
import argparse
import glob
import json
import os

import numpy as np

from firstp_sampling import DLAB, PARTS, QLAB, RECIPE, SUP

PL = ["U", "V", "Y"]
HERE = os.path.dirname(os.path.abspath(__file__))


def odds(r):
    return r / (1 - r) if 0 < r < 1 else (np.inf if r >= 1 else 0.0)


def cell(nt, ct, nd, cd):
    rt, rd = (ct / nt if nt else np.nan), (cd / nd if nd else np.nan)
    orr = odds(rt) / odds(rd) if nt and nd and 0 < rd < 1 else np.nan
    return dict(n_train=int(nt), n_train_charged=int(ct), r_train=rt, n_deploy=int(nd), n_deploy_charged=int(cd),
                r_deploy=rd, odds_ratio=orr, log_odds_ratio=float(np.log(orr)) if orr and np.isfinite(orr) and orr > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--draws", type=int, default=4, help="as firstp_sampling.py --draws")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "firstp_sampling"))
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.inp, "ev_*.npz")))
    Z = [np.load(f) for f in files]
    n = len(Z)
    S = lambda k: sum(z[k] for z in Z)
    out = dict(meta=dict(
        n_events=n, draws=a.draws, events=[os.path.basename(f)[3:-4] for f in files], recipe=RECIPE,
        definitions=dict(
            support="truth used BOTH to draw the rows and to call a cell charged: 'hits' = ost61/denoise_truth "
                    "(25 e- cut per group x wire x tick sample), 'complete' = denoise_truth_complete (no cut)",
            charged="q > 0 under the support truth (the presence target is y = log1p(q/Q0) > 0)",
            deploy="every fine cell (2 wires x 16 ticks) covered by a token in any band (cell_inputs idx >= 0), "
                   "i.e. what the deployed model is queried on",
            train="unique token-covered rows of DenoiseEvents.rows with the checkpoints' recipe, pooled over draws",
            stratifier="'hd' = density from the HITS truth, 'cd' = from the COMPLETE truth: charge (e-) within +-8 fine "
                       "cells (17x17 cells = 34 wires x 272 ticks), the cell excluded, binned as dense/analyze.py",
            r_train="charged rows / rows in the stratum", r_deploy="charged covered cells / covered cells in the stratum",
            odds_ratio="[r_train/(1-r_train)] / [r_deploy/(1-r_deploy)]: the factor by which the training prior "
                       "inflates the odds of 'charged'; a calibrated-under-training presence logit is high by log(OR)",
            key_path="r[support][stratifier][plane U|V|Y|all][density bin|all]")))
    # ---- (i) ratios
    r = {}
    for s in SUP:
        r[s] = {}
        for si, st in enumerate(("hd", "cd")):
            T = S(f"train_{s}")[si]                       # [P][D][part|all][charged]
            Dp = S(f"deploy_{st}")                        # [P][D][ch_hits][ch_complete]
            Dc = Dp.sum(3) if s == "hits" else Dp.sum(2)  # [P][D][charged under support]
            r[s][st] = {}
            for pi, pl in enumerate(PL + ["all"]):
                sel = slice(None) if pl == "all" else pi
                Tp = T[sel].reshape(-1, len(DLAB), len(PARTS) + 1, 2).sum(0) if pl == "all" else T[pi]
                Dpp = Dc.reshape(-1, len(DLAB), 2).sum(0) if pl == "all" else Dc[pi]
                d = {}
                for di, dl in enumerate(DLAB + ["all"]):
                    tt = Tp[:, -1].sum(0) if dl == "all" else Tp[di, -1]
                    dd = Dpp.sum(0) if dl == "all" else Dpp[di]
                    d[dl] = cell(tt.sum(), tt[1], dd.sum(), dd[1])
                r[s][st][pl] = d
    out["r"] = r
    # composition of training rows by rule (per event, per draw), and charged share per rule
    comp = {}
    for s in SUP:
        T = S(f"train_{s}")[0].sum((0, 1))               # [part|all][charged]
        comp[s] = {p: dict(rows_per_event_draw=float(T[i].sum()) / n / a.draws, charged_frac=float(T[i, 1] / max(T[i].sum(), 1)))
                   for i, p in enumerate(PARTS + ["all"])}
    # draws: rows recorded per draw; parts_ counts sum over draws -> per-draw normalisation
    out["composition"] = comp
    # ---- (ii) charge of trained charged rows (sig rule, after the 20000 cap)
    q = {}
    for s in SUP:
        q[s] = dict(
            charged_cells_per_event_before_cap=float(S(f"nsig_pre_{s}")) / n,
            events_hitting_cap=int(sum(int(z[f"nsig_pre_{s}"]) > RECIPE["sig_cap"] for z in Z)),
            truth_cells_per_event_by_charge=dict(zip(QLAB, (S(f"truth_qhist_{s}") / n).tolist())),
            covered_truth_cells_per_event_by_charge=dict(zip(QLAB, (S(f"truth_qhist_cov_{s}") / n).tolist())),
            trained_sig_rows_per_event_draw_by_charge=dict(zip(QLAB, (S(f"qtrain_{s}") / n / a.draws).tolist())),
            trained_sig_rows_by_their_hits_charge=dict(zip(["hits=0"] + QLAB, (S(f"qtrain_hitsq_{s}") / n / a.draws).tolist())),
            truth_charge_covered_frac=float(S(f"truth_cov_{s}")[3] / S(f"truth_cov_{s}")[2]),
            truth_cells_covered_frac=float(S(f"truth_cov_{s}")[1] / S(f"truth_cov_{s}")[0]))
    out["charge"] = q
    # ---- (iii) noise windows
    W = np.concatenate([z["windist"] for z in Z])
    cols = ["support", "kind", "plane", "d_support", "d_hits", "d_complete_gt0", "d_complete_gt25", "d_complete_gt102",
            "d_complete_gt500", "qmax_hits_inside", "qmax_complete_inside"]
    win = {}
    for si, s in enumerate(SUP):
        win[s] = dict(
            win_drawn_per_event=float(S(f"nwin_{s}")) / n, win_near_drawn_per_event=float(S(f"nwin_near_{s}")) / n,
            win_candidates_distinct_per_event=float(S(f"wincand_{s}")[0]) / n,
            win_candidate_centres_accepted_frac=float(S(f"wincand_{s}")[1] / S(f"wincand_{s}")[2]),
            win_near_candidate_origins_per_event=float(S(f"nearcand_{s}").sum()) / n)
        for ki, kind in enumerate(("win", "win_near")):
            m = (W[:, 0] == si) & (W[:, 1] == ki)
            w = W[m]
            dd = {}
            for j, c in enumerate(cols[3:9], 3):
                x = w[:, j][np.isfinite(w[:, j])]
                dd[c] = dict(p10=float(np.percentile(x, 10)), median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                             frac_zero=float((x == 0).mean())) if len(x) else None
            dd["n_windows"] = int(m.sum())
            dd["frac_with_complete_charge_inside"] = float((w[:, 10] > 0).mean()) if len(w) else None
            dd["frac_with_complete_gt25_inside"] = float((w[:, 10] > 25).mean()) if len(w) else None
            dd["frac_with_complete_gt102_inside"] = float((w[:, 10] > 102).mean()) if len(w) else None
            win[s][kind] = dd
    out["windows"] = win
    out["windows_note"] = ("distances: chessboard, fine cells (1 = 2 wires or 16 ticks), from the window box (16x8 cells) "
                           "to the nearest cell with charge above the threshold; d=0 means charge inside the window")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"), indent=1, default=float)

    # ---- markdown
    L = [f"# First-principles A: case-control sampling ratios ({n} train events, 8 runs)", "",
         "Generated by `firstp_sampling.py` + `firstp_sampling_sum.py`; full numbers in `firstp_sampling.json` "
         "(key path `r[support][stratifier][plane][density]`, definitions in `meta.definitions`). All numbers [V] "
         "(measured on these events with the checkpoints' exact recipe; rows asserted identical to `DenoiseEvents.rows`).", ""]
    for st, stl in (("hd", "density from the HITS truth"), ("cd", "density from the COMPLETE truth")):
        L += [f"## r_train vs r_deploy, all planes, stratified by {stl}", "",
              "| density (e-, +-8 cells) | hits: r_train | hits: r_deploy | hits: OR | complete: r_train | complete: r_deploy | complete: OR | deploy cells/event |",
              "|---|---|---|---|---|---|---|---|"]
        for dl in DLAB + ["all"]:
            h, c = r["hits"][st]["all"][dl], r["complete"][st]["all"][dl]
            L.append(f"| {dl} | {h['r_train']:.4f} | {h['r_deploy']:.5f} | {h['odds_ratio']:.1f} | {c['r_train']:.4f} | "
                     f"{c['r_deploy']:.5f} | {c['odds_ratio']:.1f} | {h['n_deploy'] / n:,.0f} |")
        L.append("")
    L += ["## Odds ratio per plane (hits-density strata)", "",
          "| density | " + " | ".join(f"hits {p}" for p in PL) + " | " + " | ".join(f"complete {p}" for p in PL) + " |",
          "|---|" + "---|" * 6]
    for dl in DLAB + ["all"]:
        L.append(f"| {dl} | " + " | ".join(f"{r['hits']['hd'][p][dl]['odds_ratio']:.1f}" for p in PL) + " | "
                 + " | ".join(f"{r['complete']['hd'][p][dl]['odds_ratio']:.1f}" for p in PL) + " |")
    L += ["", "## Training rows by rule (per event and draw)", "",
          "| rule | hits rows | hits charged frac | complete rows | complete charged frac |", "|---|---|---|---|---|"]
    for p in PARTS + ["all"]:
        h, c = comp["hits"][p], comp["complete"][p]
        L.append(f"| {p} | {h['rows_per_event_draw']:,.0f} | {h['charged_frac']:.3f} | {c['rows_per_event_draw']:,.0f} | {c['charged_frac']:.3f} |")
    L += ["", "(a row produced by several rules counts once per rule; 'all' counts it once.)", "",
          "## Charged cells: present vs trained (sig rule, after sig_cap = 20000)", "",
          f"hits: {q['hits']['charged_cells_per_event_before_cap']:,.0f} charged cells/event, {q['hits']['events_hitting_cap']}/{n} events over the cap; "
          f"complete: {q['complete']['charged_cells_per_event_before_cap']:,.0f}/event, {q['complete']['events_hitting_cap']}/{n} over the cap.", "",
          "| cell charge (e-) | hits: present/event | hits: trained/event/draw | complete: present/event | complete: trained/event/draw |",
          "|---|---|---|---|---|"]
    for ql in QLAB:
        L.append(f"| {ql} | {q['hits']['truth_cells_per_event_by_charge'][ql]:,.0f} | {q['hits']['trained_sig_rows_per_event_draw_by_charge'][ql]:,.0f} | "
                 f"{q['complete']['truth_cells_per_event_by_charge'][ql]:,.0f} | {q['complete']['trained_sig_rows_per_event_draw_by_charge'][ql]:,.0f} |")
    L += ["", "## Noise windows (win: 48/event far from charge; win_near: 24/event near charge)", "",
          "| | hits support | complete support |", "|---|---|---|"]
    for k in ("win_drawn_per_event", "win_candidates_distinct_per_event", "win_candidate_centres_accepted_frac",
              "win_near_drawn_per_event", "win_near_candidate_origins_per_event"):
        L.append(f"| {k} | {win['hits'][k]:,.3f} | {win['complete'][k]:,.3f} |")
    for kind in ("win", "win_near"):
        for c in ("d_support", "d_hits", "d_complete_gt0", "d_complete_gt102"):
            h, cc = win["hits"][kind][c], win["complete"][kind][c]
            L.append(f"| {kind} {c} median [p10, p90] | {h['median']:.0f} [{h['p10']:.0f}, {h['p90']:.0f}] | {cc['median']:.0f} [{cc['p10']:.0f}, {cc['p90']:.0f}] |")
        L.append(f"| {kind} frac with complete charge > 0 / > 25 / > 102 e- inside | "
                 f"{win['hits'][kind]['frac_with_complete_charge_inside']:.3f} / {win['hits'][kind]['frac_with_complete_gt25_inside']:.3f} / {win['hits'][kind]['frac_with_complete_gt102_inside']:.3f} | "
                 f"{win['complete'][kind]['frac_with_complete_charge_inside']:.3f} / {win['complete'][kind]['frac_with_complete_gt25_inside']:.3f} / {win['complete'][kind]['frac_with_complete_gt102_inside']:.3f} |")
    L += ["", out["windows_note"] + "."]
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
