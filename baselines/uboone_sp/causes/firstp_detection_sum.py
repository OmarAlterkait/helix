#!/usr/bin/env python3
"""Write causes/results/firstp_detection.md from firstp_detection.json, firstp_cut.json, hitless.json
and firstp_sampling.json (all [V]: computed by the firstp_* scripts named in each section)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
FP = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/firstp"
K = "floor_pp_eff1pct_0.1-0.2"
KN = "near_pp_eff1pct_0.1-0.2"
MOD = ["M4n", "M6d", "M6dc", "lb_dec_hits", "lb_dec_ct", "lb_dec_cond_hits", "lb_dec_cond_ct"]
SC = ["_charge", "_pres", "_gated"]
PUB, NOI, ISO, NIS = ("hits (as published)", "no own deposit >= 0.02 MeV", "isolated positives (ring max <= 25 e-)",
                      "non-isolated positives (ring max > 25 e-)")


def f3(x):
    return "--" if x is None or x != x else f"{x:.3f}"


def iv(b, m, k):
    try:
        lo, hi = b[m][k]
        return f"[{lo:.3f}, {hi:.3f}]"
    except (KeyError, TypeError):
        return ""


def main():
    D = json.load(open(os.path.join(R, "firstp_detection.json")))
    C = json.load(open(os.path.join(R, "firstp_cut.json")))
    H = json.load(open(os.path.join(FP, "hitless.json")))
    S = json.load(open(os.path.join(R, "firstp_sampling.json")))
    P, B, DM = D["point"], D["bootstrap"], D["draws_mean"]
    L = ["# First principles: sampling prior, the 25 e- cut, and deposit-level detection", "",
         "Post-hoc diagnostics only (no training, no new losses or heads). Every number is [V] (computed from the files "
         "named; scripts in `baselines/uboone_sp/causes/firstp_*.py`, outputs under `helix_runs/firstp/`) unless "
         "marked [A]. Intervals: 16-84 % over 200 event-bootstrap resamples of the 128 test events (ev260-387).", ""]
    # ---------------------------------------------------------------- A summary
    r = S["r"]
    L += ["## A. The training prior (details: `firstp_sampling.md`, `firstp_sampling.json`)", "",
          f"{S['meta']['n_events']} train events (25 per run, shard 0010, 8 runs), {S['meta']['draws']} draws each, with the "
          "checkpoints' exact recipe (identical `meta['args']` for M6d, M6dc and the four lb_* arms: sig_cap 20000, "
          "near 1, neg 1, cov 2, win 48, near_any 1, win_near 24; rows asserted equal to `DenoiseEvents.rows`).", "",
          "| density (hits truth, +-8 cells) | hits r_train | hits r_deploy | hits OR | complete r_train | complete r_deploy | complete OR |",
          "|---|---|---|---|---|---|---|"]
    for dl in ["iso", "<=10k", "10k-100k", "100k-1M", "1M-4M", ">4M", "all"]:
        h, c = r["hits"]["hd"]["all"][dl], r["complete"]["hd"]["all"][dl]
        L.append(f"| {dl} | {h['r_train']:.4f} | {h['r_deploy']:.5f} | {h['odds_ratio']:.1f} | {c['r_train']:.4f} | {c['r_deploy']:.5f} | {c['odds_ratio']:.1f} |")
    q = S["charge"]
    L += ["", f"* The presence target is trained at odds {r['hits']['hd']['all']['all']['odds_ratio']:.1f}x the deployment odds "
          f"overall (hits), up to ~16x where activity is sparse and ~1.5x in the densest cells: a presence probability "
          f"calibrated on training rows over-states P(charge) by log(OR) = 0.4-2.8 nats depending on density, i.e. the "
          f"operating point is density-dependent [V]. The complete truth lowers the sparse-region OR to ~9.5 because its "
          f"charged population is 1.8x larger while the row budget is fixed [V for the ORs; A for the reason].",
          f"* Complete truth: {q['complete']['charged_cells_per_event_before_cap']:,.0f} charged cells/event vs "
          f"{q['hits']['charged_cells_per_event_before_cap']:,.0f} (hits); "
          f"{sum(q['complete']['truth_cells_per_event_by_charge'][k] for k in ('<1e-3', '1e-3-1', '1-5', '5-25')):,.0f}/event are below 25 e- "
          f"and {q['complete']['truth_cells_per_event_by_charge']['<1e-3']:,.0f}/event below 1e-3 e-, all of them presence positives "
          f"(y > 0) [V]. With sig_cap = 20000 binding in {q['complete']['events_hitting_cap']}/{S['meta']['n_events']} events "
          f"(hits: {q['hits']['events_hitting_cap']}), the uniform cap spends ~38 % of the charged-row budget on sub-25 e- cells, so "
          f"bright cells (> 16k e-) trained per event fall from {q['hits']['trained_sig_rows_per_event_draw_by_charge']['16-64k'] + q['hits']['trained_sig_rows_per_event_draw_by_charge']['>64k']:,.0f} "
          f"to {q['complete']['trained_sig_rows_per_event_draw_by_charge']['16-64k'] + q['complete']['trained_sig_rows_per_event_draw_by_charge']['>64k']:,.0f} (-42 %) [V].",
          "* The training noise windows are unaffected: under either truth, 0.000 (win) / <= 0.001 (win_near) of them hold any "
          "complete charge; median distance to activity 29 fine cells (win) and 5 (win_near) [V].", ""]
    # ---------------------------------------------------------------- B
    T = C["train_events"]; M = C["test_events_mechanism"]
    L += ["## B. The 25 e- cut is per (group, wire, tick) sample, not per cell (`firstp_cut.py`, `firstp_cut.json`)", "",
          f"Train events ({C['meta']['n_events']}): of the cells with complete charge > 102 e-, "
          f"{T['>102']['all']['frac_hits0']:.2%} have hits label 0 and {T['>102']['all']['frac_hits_lt50']:.2%} have < 50 % of their charge; "
          f"for > 500 e- it is {T['>500']['all']['frac_hits0']:.3%} and {T['>500']['all']['frac_hits_lt50']:.2%} [V].", "",
          "| density (complete, +-8 cells) | cells > 102 e- | hits = 0 | hits < 50 % | median hits/complete | charge lost |", "|---|---|---|---|---|---|"]
    for dl, v in T[">102"]["by_density"].items():
        if v.get("n"):
            L.append(f"| {dl} | {v['n']:,} | {v['frac_hits0']:.2%} | {v['frac_hits_lt50']:.2%} | {v['median_ratio']:.3f} | {v['charge_lost']:.2%} |")
    L += ["", "| cell charge (e-) | cells | hits = 0 | hits < 50 % | median hits/complete | charge lost |", "|---|---|---|---|---|---|"]
    for ql, v in T["by_charge"].items():
        if v.get("n"):
            L.append(f"| {ql} | {v['n']:,} | {v['frac_hits0']:.2%} | {v['frac_hits_lt50']:.2%} | {v['median_ratio']:.3f} | {v['charge_lost']:.2%} |")
    L += ["", f"Mechanism, test events (re-simulated with group identity; the cut reproduces the hits truth by construction, "
          f"validated at rel. L2 0.4 % in the corpus build): cells > 102 e- by their LARGEST single per-group sample and by "
          f"the number of deposit groups sharing them [V]:", "",
          "| largest per-group sample (e-) | cells | hits = 0 | hits < 50 % | charge lost |", "|---|---|---|---|---|"]
    for sl, v in M[">102"]["by_max_sample"].items():
        if v.get("n"):
            L.append(f"| {sl} | {v['n']:,} | {v['frac_hits0']:.2%} | {v['frac_hits_lt50']:.2%} | {v['charge_lost']:.2%} |")
    L += ["", "| groups in the cell | cells | hits = 0 | hits < 50 % | charge lost |", "|---|---|---|---|---|"]
    for nl, v in M[">102"]["by_groups"].items():
        if v.get("n"):
            L.append(f"| {nl} | {v['n']:,} | {v['frac_hits0']:.2%} | {v['frac_hits_lt50']:.2%} | {v['charge_lost']:.2%} |")
    L += ["", "Reading: the old label of a cell is a function of how its charge is SPLIT (over 32 samples per group and over "
          "groups), not of its charge. A cell is zeroed whenever no single group puts > 25 e- in any one of its samples, "
          "whatever its total (up to ~800 e- for one group spread evenly) [V for the counts, A for the 800 e- bound = 32 x 25]. "
          "The label noise is set by the cell's charge: among 102-250 e- cells 6.7 % are zeroed, 25 % keep < 50 % and 35 % of "
          "their charge is dropped; 250-500 e-: 0.6 % / 5.7 % / 19 %; above 1k e- nothing is zeroed but 1-7 % of the charge "
          "is still dropped [V]. At fixed charge threshold it rises with the number of sharing groups (0.5 % -> 1.8 % zeroed "
          "from 1 to 5-8 groups) and the dropped share is largest at low density (3.9 % of > 102 e- charge where the "
          "+-8-cell neighbourhood holds < 10k e-) [V]. So the hits label is noisy for faint charge (a 100-250 e- cell is "
          "labelled at 0.69 of its charge on median, and 0 one time in fifteen), grouping-dependent, and only mildly "
          "affected in dense charge, where many groups each exceed 25 e- [V].", ""]
    # ---------------------------------------------------------------- C
    nc = D["noise_window_contents"]; iw = D["iso_windows"]
    L += ["## C. Deposit-level detection (`firstp_windows.py`, `firstp_detection.py`, `firstp_hitless.py`)", "",
          "### C1. The floor's noise windows under the complete truth, with particle identity", "",
          "| | bg (far), n | any complete charge | max cell > 25 / 102 / 500 e- | own particle (f_in >= 0.5) | from own / from tails (> 25 e-) |",
          "|---|---|---|---|---|---|"]
    for kind in ("bg", "bgn"):
        for pl in ("all", "U", "V", "Y"):
            v = nc[kind].get(pl)
            if not v:
                continue
            L.append(f"| {kind} {pl} | {v['n']} | {v['frac_any_complete_charge']:.4f} | {v['frac_max_cell_gt25']:.4f} / "
                     f"{v['frac_max_cell_gt102']:.4f} / {v['frac_max_cell_gt500']:.4f} | {v['frac_with_own_particle']:.4f} | "
                     f"{v['frac_max_cell_gt25_from_own']:.4f} / {v['frac_max_cell_gt25_from_tails']:.4f} |")
    own = nc["bgn"]["all"].get("own_E_quantiles_if_own")
    L += ["", f"* The noise windows ARE empty under the complete truth: 1 of 7680 far windows holds any charge (4e-7 e-), "
          f"none has an own particle; {nc['bgn']['all']['frac_any_complete_charge']:.2%} of near windows hold charge, "
          f"{nc['bgn']['all']['frac_max_cell_gt102']:.3%} above 102 e-, and {nc['bgn']['all']['frac_with_own_particle']:.3%} "
          f"({round(nc['bgn']['all']['frac_with_own_particle'] * 7680)} windows) contain an own particle, all of "
          f"{own[0]:.3f}-{own[-1]:.3f} MeV [V]. "
          "Diffusion tails reach ~2 wires / ~6 ticks beyond a hits pixel (K_w = 2, K_t = 6 in JAXTPC), far less than the "
          "16 wire / 128 tick padding of a bg window or the 1-cell margin of a bgn window [A, from the kernel extent].",
          f"* Completeness of the complete truth itself: particles with NO hits are left out (out of window). Re-simulated at an "
          f"in-window t0, {H['ambiguous']['n_per_event']:,.0f} of them per event would ALSO leave no hit (largest sample <= 25 e-) and could "
          f"therefore be in the window, unseen by both truths; together they carry {H['ambiguous']['charge_per_event']:,.0f} e- per event "
          f"(median {H['ambiguous']['Q_quantiles_e'][0]:.0f} e-, 99.9 % below {H['ambiguous']['Q_quantiles_e'][3]:.0f} e- per particle) [V] "
          "-- below the single-wire noise and irrelevant for detection [A]. Their step energy is large for their charge "
          f"({H['ambiguous']['energy_per_event_MeV']:.1f} MeV/event; heavily quenched deposits) [V].", "",
          "**E_min.** Since no far window and 6 of 7680 near windows contain an own deposit (all < 0.02 MeV), every E_min in "
          "[0, 0.1] MeV gives the same noise set; E_min = 0.02 MeV is the primary choice because isolated deposits below it "
          "are not detected at all at 1 % FPR (0/5; 0.02-0.05 MeV: 2-12 % of 41, C3), i.e. they are noise-like and an own "
          "deposit that small does not make a window signal [V]. "
          "The published floor and near efficiencies are therefore UNCHANGED by the deposit-level noise definition, for every "
          "model (table C2) [V].", "",
          "### C2. Detection at 1 % FPR per plane, 0.1-0.2 MeV (stored corpus)", "",
          f"Positives: the `iso` windows ({iw['n']} in total, 1522 at 0.1-0.2 MeV). Their surroundings: "
          f"{iw['frac_ring_max_gt25']:.1%} have complete charge > 25 e- in the same 16-wire / 128-tick padding a bg window "
          f"must keep empty ({iw['frac_ring_max_gt2043']:.1%} > 1 Q0; median ring charge {iw['ring_q_quantiles'][2]:,.0f} e-) [V]: "
          "the iso deposits are isolated from FOREIGN CHARGE IN THE WINDOW (< 5 %), not from activity. 'Isolated positives' "
          "keep only those whose padding holds no cell > 25 e- (107 at 0.1-0.2 MeV), the deposit-level analogue of a bg window.", "",
          "| model | published (far) | noise = no own deposit >= 0.02 MeV | isolated positives | non-isolated positives | published (near) | isolated (near) |",
          "|---|---|---|---|---|---|---|"]
    for m in MOD:
        for s in SC:
            k = m + s
            if k not in P[PUB]:
                continue
            L.append(f"| {k} | {f3(P[PUB][k].get(K))} {iv(B[PUB], k, K)} | {f3(P[NOI][k].get(K))} | "
                     f"{f3(P[ISO][k].get(K))} {iv(B[ISO], k, K)} | {f3(P[NIS][k].get(K))} | {f3(P[PUB][k].get(KN))} | {f3(P[ISO][k].get(KN))} |")
    L += ["", "Per plane, published vs isolated positives (far threshold, 0.1-0.2 MeV):", "",
          "| model | U pub | U iso | V pub | V iso | Y pub | Y iso |", "|---|---|---|---|---|---|---|"]
    for m in ("M4n", "M6d", "M6dc", "lb_dec_hits", "lb_dec_ct"):
        for s in ("_pres", "_gated"):
            k = m + s
            if k in P[PUB]:
                L.append(f"| {k} | " + " | ".join(f"{f3(P[PUB][k].get(K + '_' + p))} | {f3(P[ISO][k].get(K + '_' + p))}" for p in "UVY") + " |")
    # efficiency vs fine E
    L += ["", "### C3. Efficiency vs deposit energy (far threshold): the low-energy 'detections' are activity nearby", "",
          "| model | positives | 0.01-0.02 | 0.02-0.05 | 0.05-0.1 | 0.1-0.2 MeV |", "|---|---|---|---|---|---|"]
    for k in ("M6d_pres", "M6d_gated", "lb_dec_hits_pres", "lb_dec_ct_pres", "lb_dec_hits_gated", "lb_dec_ct_gated"):
        for nm, dn in (("all (published)", PUB), ("isolated", ISO)):
            v = P[dn].get(k, {})
            L.append(f"| {k} | {nm} | " + " | ".join(
                f"{f3(v.get('floor_effE_' + e))} (n={v.get('floor_nE_' + e, 0)})" for e in ("0.01-0.02", "0.02-0.05", "0.05-0.1", "0.1-0.2")) + " |")
    L += ["", "A 0.01-0.02 MeV deposit (~300-600 e- before response, at or below one wire's noise) is 'detected' 20-37 % of "
          "the time at a 1 % false-positive rate when it sits near activity, and 0 % (0/5) when it is isolated [V]: the far "
          "threshold credits the score that nearby activity lends the window (haze, or the decoder's receptive field), "
          "not the deposit [A for the mechanism; V for the numbers]. The near-window threshold (bgn) removes most of it "
          "(near efficiencies at 0.01-0.05 MeV are 1-8 %) [V].", ""]
    # gains
    L += ["### C4. The label arms' gain (complete-truth label minus hits label), paired bootstrap", "",
          "| pair | score | published (far) | noise redefined | isolated positives (far) | non-isolated (far) | published (near) | isolated (near) | draws-mean far: published / isolated |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ct, h in (("lb_dec_ct", "lb_dec_hits"), ("lb_dec_cond_ct", "lb_dec_cond_hits")):
        for s in SC:
            def g(dn, key=K):
                a_, b_ = P[dn].get(ct + s, {}).get(key), P[dn].get(h + s, {}).get(key)
                return None if a_ is None or b_ is None else a_ - b_

            def gi(dn, key=K):
                try:
                    lo, hi = B[dn]["gain"][f"{ct}-{h}{s}:{key}"]
                    return f" [{lo:+.3f}, {hi:+.3f}]"
                except KeyError:
                    return ""

            def dmg(dn):
                try:
                    return f"{DM[dn][ct + s][K] - DM[dn][h + s][K]:+.3f}"
                except KeyError:
                    return "--"
            L.append(f"| {ct} - {h} | {s[1:]} | {g(PUB):+.3f}{gi(PUB)} | {g(NOI):+.3f} | {g(ISO):+.3f}{gi(ISO)} | "
                     f"{g(NIS):+.3f}{gi(NIS)} | {g(PUB, KN):+.3f}{gi(PUB, KN)} | {g(ISO, KN):+.3f}{gi(ISO, KN)} | {dmg(PUB)} / {dmg(ISO)} |")
    L += ["", "### Verdict", "",
          "1. **Noise windows (C1):** under the complete truth the floor's far and near noise windows are empty of charge "
          "and of own deposits (no far window, 0.08 % of near windows with a 0.01-0.02 MeV deposit). Redefining 'noise' as "
          "'no own deposit above E_min' changes no number: 100 % of every detection result, and of the label arms' gain, "
          "survives THIS change [V].",
          "2. **Positives (C2-C3):** the real ambiguity is on the positive side. 93 % of the `iso` positives sit within "
          "16 wires / 128 ticks of other charge, while the noise windows by construction do not; at the far threshold a "
          "window's score includes what nearby activity lends it. On deposit-level positives (as empty around as a noise "
          "window) every useful score loses 0.10-0.19 of efficiency at 0.1-0.2 MeV (M6d pres 0.842 -> 0.692, M4n pres 0.851 -> "
          "0.692), with only 107 such windows (+-0.045) [V]. The ranking M4n ~ M6d ~ M6dc > label arms is preserved [V].",
          "3. **Label arms (C4):** the published far-threshold gain of lb_dec_ct over lb_dec_hits (+0.041 pres, +0.037 "
          "gated, stored corpus) (a) shrinks to +0.009 / +0.016 as a mean over the five fresh noise draws, before any "
          "redefinition [V]; (b) is entirely carried by the non-isolated positives (+0.045 / +0.040) and is -0.009 / +0.000 "
          "(intervals spanning 0) on isolated ones [V]; (c) is largest at 0.01-0.05 MeV (+0.15-0.18 on all positives), "
          "where isolated deposits are detected at <= 12 % by every model [V]. So the far-threshold gain is consistent "
          "with the complete label teaching presence in the diffuse surroundings of activity, which that threshold credits, "
          "not with better detection of an isolated deposit [A: inference; the 107-window sample cannot exclude a true "
          "gain of ~0.04]. What does survive is the NEAR-threshold gain of the charge and gated outputs (+0.12 published; "
          "on isolated positives +0.09 charge / +0.08 gated): with the hits label those outputs leave a haze next to "
          "activity that the near noise windows punish, and the complete label removes part of it [V for the numbers; A "
          "for the reading]. The cond arms lose with the complete label at the far threshold (-0.02 to -0.16) [V].",
          "4. **Sampling (A) and the cut (B):** the presence head is trained at 9.5x the deployment odds (16x in sparse "
          "regions); switching to the complete label adds 31k sub-25 e- positives per event, lowers that ratio in sparse "
          "regions to ~9.5x and, through the fixed sig_cap, removes 42 % of the bright training cells. The hits label's "
          "own noise is concentrated in faint cells (102-250 e-: 6.7 % zeroed, 35 % of the charge dropped) and grows with "
          "the number of deposit groups sharing a cell [V].", ""]
    open(os.path.join(R, "firstp_detection.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
