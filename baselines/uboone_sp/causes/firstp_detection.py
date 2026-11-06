#!/usr/bin/env python3
"""First-principles C: are the floor evaluation's noise windows empty under the complete truth,
and how much of the detection numbers (and the label arms' gain) survives a deposit-level definition?

    scripts/helix_run.sh python baselines/uboone_sp/causes/firstp_detection.py \
        --windows /global/cfs/cdirs/m5238/users/oalter/helix_runs/firstp/windows

Inputs: firstp_windows.py output (complete charge with particle identity on every window of
the 128 test events) and the stored per-window scores of every model
(helix_runs/denoise/results/dn_{full,label}_<tag>[_pres|_gated]_trained_windows.json, written by
scripts/eval_denoise.py; ``@d<k>`` = fresh noise draw k).

Noise windows of the floor: ``bg`` (far: no hits pixel within 16 wires / 128 ticks) and ``bgn``
(near: no hits cell within 1 cell, some within 8). A particle is a window's OWN deposit when at
least half of its complete charge on that plane lies inside the window (f_in >= 0.5) -- the iso
windows' own criterion is f_in >= 0.9, so this counts anything that is mostly there; the rest is
a TAIL of a neighbour. Deposit-level noise window (E_min): no own particle with deposited energy
>= E_min. Detection is re-scored exactly as eval_denoise.pp_scalars (per-plane 99% quantile of
the noise windows' scores, efficiency of iso windows at that threshold), with the event bootstrap
(200 resamples, 16-84 %) and paired differences for the label-arm gains.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
RES = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/denoise/results"
COMPLETE = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
MODELS = [("M4n", "dn_full_M4n_ft_full_near"), ("M6d", "dn_full_M6d_ft_full_dec"),
          ("M6dc", "dn_full_M6dc_ft_full_dec_cond"), ("lb_dec_hits", "dn_label_lb_dec_hits"),
          ("lb_dec_cond_hits", "dn_label_lb_dec_cond_hits"), ("lb_dec_ct", "dn_label_lb_dec_ct"),
          ("lb_dec_cond_ct", "dn_label_lb_dec_cond_ct")]
SCORES = ["", "_pres", "_gated"]
KINDS = ["iso", "pair", "bg", "bgn"]
EB = (0.0, 0.1, 0.2, 0.5, 1.0, 1e9)
SNR = ((0, 1e-9), (1e-9, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 1e9))
EMINS = [0.0, 0.003, 0.01, 0.02, 0.05, 0.1]
PRIMARY = 0.02
NB = 200


def ring_stats(CK, CQ, g, w0, t0):
    """Complete charge around a window: the padded box the bg windows must keep free of hits
    (16 wires / 128 ticks on every side), the window itself excluded -> (ring sum, ring max cell)."""
    from helix.probe.resolution import fkey
    fw = np.arange((w0 - 16) // 2, (w0 + 48) // 2); ft = np.arange((t0 - 128) // 16, (t0 + 256) // 16)
    FWm, FTm = np.meshgrid(fw, ft, indexing="ij")
    inside = (FWm >= w0 // 2) & (FWm < w0 // 2 + 16) & (FTm >= t0 // 16) & (FTm < t0 // 16 + 8)
    ok = (FWm >= 0) & ~inside
    k = fkey(np.full(int(ok.sum()), g), FWm[ok], FTm[ok])
    if not len(CK):
        return 0.0, 0.0
    j = np.clip(np.searchsorted(CK, k), 0, len(CK) - 1)
    q = np.where(CK[j] == k, CQ[j], 0.0)
    return float(q.sum()), float(q.max()) if len(q) else 0.0


def load_windows(d):
    """{(ev, kind, g, w0, t0): dict(qc_tot, qc_max, own_E (max E of own particles, -1 none),
    own_q, tail_q, tail_max, n_own, E (iso))}"""
    W = {}
    for f in sorted(glob.glob(os.path.join(d, "ev*.npz"))):
        ev = int(os.path.basename(f)[2:5])
        z = np.load(f)
        wt = z["wt"]
        shard, event = str(z["shard"]), int(z["event"])
        run = os.path.basename(os.path.dirname(shard)); tag = os.path.basename(shard)[-7:-3]
        zc = np.load(os.path.join(COMPLETE, run, f"truth_{tag}.npz"))
        ii = int(np.nonzero(zc["event"] == event)[0][0])
        CK = zc["keys"][zc["offsets"][ii]:zc["offsets"][ii + 1]].astype(np.int64)
        CQ = zc["q"][zc["offsets"][ii]:zc["offsets"][ii + 1]].astype(np.float64)
        for i in range(len(z["win_kind"])):
            r = wt[wt[:, 0] == i] if len(wt) else np.zeros((0, 8))
            fin = r[:, 2] / np.maximum(r[:, 3], 1e-30)
            own = fin >= 0.5
            cen = (r[:, 5] >= 0) & (r[:, 5] < 32) & (r[:, 6] >= 0) & (r[:, 6] < 128)
            W[(ev, KINDS[int(z["win_kind"][i])], int(z["win_g"][i]), int(z["win_w0"][i]), int(z["win_t0"][i]))] = dict(
                qc_tot=float(z["wc"][i, 0]), qc_max=float(z["wc"][i, 1]), qh=float(z["win_qhits"][i]),
                own_E=float(r[own, 4].max()) if own.any() else -1.0, own_q=float(r[own, 2].sum()),
                own_max=float(r[own, 7].max()) if own.any() else 0.0,
                tail_q=float(r[~own, 2].sum()), tail_max=float(r[~own, 7].max()) if (~own).any() else 0.0,
                n_own=int(own.sum()), n_part=len(r), cen_E=float(r[cen, 4].max()) if cen.any() else -1.0,
                E=float(z["win_E"][i]),
                ring=ring_stats(CK, CQ, int(z["win_g"][i]), int(z["win_w0"][i]), int(z["win_t0"][i])))
    return W


def eff_table(ws, noise_ok, keys_eb=(1,), pos_ok=None):
    """pp_scalars on window list ``ws`` (dicts with kind, score, p, eb, snr, clean, ev) where
    noise windows are kept only if noise_ok(w). Returns {key: value}."""
    out = {}
    for lab, kind in (("floor", "bg"), ("near", "bgn")):
        thr = {}
        for p in range(3):
            s = np.array([w["score"] for w in ws if w["kind"] == kind and w["p"] == p and noise_ok(w)])
            if len(s):
                thr[p] = np.quantile(s, 0.99)
        if len(thr) < 3:
            continue
        iso = [w for w in ws if w["kind"] == "iso" and (pos_ok is None or pos_ok(w))]
        dec = np.array([w["score"] > thr[w["p"]] for w in iso])
        eb = np.array([w["eb"] for w in iso]); pp = np.array([w["p"] for w in iso])
        for i in keys_eb:
            nm = f"{EB[i]:g}-{EB[i + 1]:g}"
            m = eb == i
            if m.any():
                out[f"{lab}_pp_eff1pct_{nm}"] = float(dec[m].mean())
            for p in range(3):
                mp = m & (pp == p)
                if mp.any():
                    out[f"{lab}_pp_eff1pct_{nm}_{'UVY'[p]}"] = float(dec[mp].mean())
        snr = np.array([w.get("snr", -1) for w in iso]); cl = np.array([bool(w.get("clean")) for w in iso])
        for lo, hi in SNR:
            m = cl & (snr >= lo) & (snr < hi)
            tag = "nosignal" if hi <= 1e-9 else f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}-inf"
            if m.any():
                out[f"snr_pp_eff1pct_{'far' if lab == 'floor' else 'near'}_{tag}"] = float(dec[m].mean())
        # efficiency in fine energy bins (E_min justification)
        E = np.array([w.get("E", np.nan) for w in iso])
        for lo, hi in ((0, 0.003), (0.003, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2)):
            m = (E >= lo) & (E < hi)
            if m.any():
                out[f"{lab}_effE_{lo:g}-{hi:g}"] = float(dec[m].mean())
                out[f"{lab}_nE_{lo:g}-{hi:g}"] = int(m.sum())
    return out


def boot(ws_by_model, noise_ok, evs, nb=NB, seed=0, pos_ok=None):
    """Bootstrap over events: per model, list of eff dicts."""
    rng = np.random.default_rng(seed)
    by_ev = {m: {} for m in ws_by_model}
    for m, ws in ws_by_model.items():
        for w in ws:
            by_ev[m].setdefault(w["ev"], []).append(w)
    out = {m: [] for m in ws_by_model}
    for _ in range(nb):
        sm = rng.choice(evs, len(evs), replace=True)
        for m in ws_by_model:
            out[m].append(eff_table([w for v in sm for w in by_ev[m].get(v, [])], noise_ok, pos_ok=pos_ok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--res", default=RES)
    ap.add_argument("--nb", type=int, default=NB)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "firstp_detection.json"))
    a = ap.parse_args()
    W = load_windows(a.windows)
    print(f"{len(W)} windows from {a.windows}", flush=True)
    out = dict(meta=dict(own="particle with >= 50% of its complete charge on that plane inside the window",
                         noise_def="noise window(E_min): bg/bgn window with no own particle of deposited energy >= E_min",
                         primary_E_min=PRIMARY, E_mins=EMINS, bootstrap=f"{a.nb} event resamples, 16-84%"))
    # ---- contents of the noise windows
    cont = {}
    for kind in ("bg", "bgn"):
        ws = [v for k, v in W.items() if k[1] == kind]
        ps = [k[2] % 3 for k in W if k[1] == kind]
        c = dict(n=len(ws))
        for nm, sel in (("all", lambda i: True), *[(p, (lambda i, p=pi: ps[i] == p)) for pi, p in enumerate("UVY")]):
            idx = [i for i in range(len(ws)) if sel(i)]
            x = [ws[i] for i in idx]
            if not x:
                continue
            qm = np.array([v["qc_max"] for v in x]); qt = np.array([v["qc_tot"] for v in x])
            ownE = np.array([v["own_E"] for v in x]); tailm = np.array([v["tail_max"] for v in x]); ownm = np.array([v["own_max"] for v in x])
            c[nm] = dict(n=len(x), frac_any_complete_charge=float((qt > 0).mean()),
                         **{f"frac_max_cell_gt{t}": float((qm > t).mean()) for t in (25, 102, 500)},
                         frac_with_own_particle=float((ownE >= 0).mean()),
                         **{f"frac_own_E_ge{e:g}": float((ownE >= e).mean()) for e in EMINS[1:]},
                         **{f"frac_max_cell_gt{t}_from_own": float((ownm > t).mean()) for t in (25, 102, 500)},
                         **{f"frac_max_cell_gt{t}_from_tails": float((tailm > t).mean()) for t in (25, 102, 500)},
                         q_total_median_if_any=float(np.median(qt[qt > 0])) if (qt > 0).any() else 0.0,
                         q_total_p90=float(np.percentile(qt, 90)), hits_charge_max=float(max(v["qh"] for v in x)),
                         own_E_quantiles_if_own=[float(np.percentile(ownE[ownE >= 0], q)) for q in (10, 50, 90, 99)]
                         if (ownE >= 0).any() else None)
        cont[kind] = c
    out["noise_window_contents"] = cont
    # iso windows under the complete truth: own charge share
    iso = [v for k, v in W.items() if k[1] == "iso"]
    rm = np.array([v["ring"][1] for v in iso]); rq = np.array([v["ring"][0] for v in iso]); Ei = np.array([v["E"] for v in iso])
    out["iso_windows"] = dict(
        n=len(iso), median_complete_over_hits=float(np.median([v["qc_tot"] / max(v["qh"], 1e-9) for v in iso])),
        ring_note="complete charge in the bg padding box (16 wires / 128 ticks each side) around the window, window excluded",
        **{f"frac_ring_max_gt{t}": float((rm > t).mean()) for t in (1, 25, 102, 500, 2043)},
        ring_q_quantiles=[float(np.percentile(rq, q)) for q in (10, 25, 50, 75, 90)],
        by_E={f"{lo:g}-{hi:g}": dict(n=int(((Ei >= lo) & (Ei < hi)).sum()),
                                      frac_ring_max_gt25=float((rm[(Ei >= lo) & (Ei < hi)] > 25).mean()) if ((Ei >= lo) & (Ei < hi)).any() else None)
              for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1e9))})
    bgr = np.array([v["ring"][1] for k, v in W.items() if k[1] == "bg"])
    out["bg_ring_check"] = dict(frac_ring_max_gt25=float((bgr > 25).mean()), note="bg windows by construction have no hits pixel in this box")

    # ---- re-score
    def load_scores(stem):
        p = os.path.join(a.res, stem + "_trained_windows.json")
        if not os.path.exists(p):
            return None
        ws = json.load(open(p))
        miss = 0
        for w in ws:
            w["p"] = int(w["g"]) % 3
            info = W.get((int(w["ev"]), w["kind"], int(w["g"]), int(w["w0"]), int(w["t0"])))
            if info is None:
                miss += 1; w["own_E"] = -1.0; w["qc_max"] = 0.0
            else:
                w["own_E"], w["qc_max"], w["E"] = info["own_E"], info["qc_max"], info["E"]
                w["ring_q"], w["ring_max"] = info["ring"]
        return ws, miss

    ALL = lambda w: True                                              # noqa: E731
    ISO25 = lambda w: w.get("ring_max", 1e9) <= 25.0                   # noqa: E731
    defs = {"hits (as published)": (ALL, ALL)}
    for e in EMINS:
        defs[f"no own deposit >= {e:g} MeV" if e > 0 else "no own particle at all"] = (
            ((lambda w, e=e: w["own_E"] < e) if e > 0 else (lambda w: w["own_E"] < 0)), ALL)
    defs["no complete cell > 102 e-"] = (lambda w: w["qc_max"] <= 102, ALL)
    # deposit-level POSITIVES: iso deposits whose surroundings are as empty as a bg window's
    defs["isolated positives (ring max <= 25 e-)"] = (ALL, ISO25)
    defs["isolated positives (ring max <= 1 e-)"] = (ALL, lambda w: w.get("ring_max", 1e9) <= 1.0)
    defs["non-isolated positives (ring max > 25 e-)"] = (ALL, lambda w: w.get("ring_max", -1) > 25.0)
    defs[f"isolated positives + no own deposit >= {PRIMARY:g} MeV"] = (lambda w: w["own_E"] < PRIMARY, ISO25)
    rows, misses = {}, {}
    variants = [(m, stem + s, s) for m, stem in MODELS for s in SCORES]
    variants += [(m + f"@d{k}", stem + f"@d{k}" + s, s) for m, stem in MODELS if m.startswith("lb_") for k in range(5) for s in SCORES]
    WS = {}
    for name, stem, s in variants:
        r = load_scores(stem)
        if r is None:
            continue
        WS[name + (s or "_charge")] = r[0]; misses[name + (s or "_charge")] = r[1]
    evs = sorted({w["ev"] for ws in WS.values() for w in ws})
    for dname, (ok, pok) in defs.items():
        rows[dname] = {k: eff_table(ws, ok, pos_ok=pok) for k, ws in WS.items()}
    out["missing_window_matches"] = misses
    out["point"] = rows
    # bootstrap (stored corpus only) for the published and the primary definition, with paired label-arm gains
    pairs = [("lb_dec_ct", "lb_dec_hits"), ("lb_dec_cond_ct", "lb_dec_cond_hits")]
    keys = ["floor_pp_eff1pct_0.1-0.2", "near_pp_eff1pct_0.1-0.2"] + [f"floor_pp_eff1pct_0.1-0.2_{p}" for p in "UVY"]
    bo = {}
    stored = {k: v for k, v in WS.items() if "@d" not in k}
    for dname in ("hits (as published)", f"no own deposit >= {PRIMARY:g} MeV", "isolated positives (ring max <= 25 e-)",
                  "non-isolated positives (ring max > 25 e-)"):
        B = boot(stored, defs[dname][0], evs, a.nb, pos_ok=defs[dname][1])
        d = {}
        for m in stored:
            d[m] = {k: [float(x) for x in np.nanpercentile([b.get(k, np.nan) for b in B[m]], [16, 84])] for k in keys}
        for ct, h in pairs:
            for s in ("_charge", "_pres", "_gated"):
                for k in keys:
                    diff = [b1.get(k, np.nan) - b2.get(k, np.nan) for b1, b2 in zip(B[ct + s], B[h + s])]
                    d.setdefault("gain", {})[f"{ct}-{h}{s}:{k}"] = [float(x) for x in np.nanpercentile(diff, [16, 84])]
        bo[dname] = d
    out["bootstrap"] = bo
    # draws mean for the label arms
    dm = {}
    for dname in defs:
        for m, _ in MODELS:
            if not m.startswith("lb_"):
                continue
            for s in ("_charge", "_pres", "_gated"):
                vals = [rows[dname].get(f"{m}@d{k}{s}") for k in range(5)]
                vals = [v for v in vals if v]
                if vals:
                    dm.setdefault(dname, {})[m + s] = {k: float(np.mean([v.get(k, np.nan) for v in vals])) for k in keys}
                    dm[dname][m + s]["draw_sd_floor"] = float(np.std([v.get(keys[0], np.nan) for v in vals]))
    out["draws_mean"] = dm
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(cont, indent=1)[:3000])
    for dname in defs:
        print(dname, {k: round(v.get("floor_pp_eff1pct_0.1-0.2", np.nan), 4) for k, v in rows[dname].items() if "@d" not in k})


if __name__ == "__main__":
    main()
