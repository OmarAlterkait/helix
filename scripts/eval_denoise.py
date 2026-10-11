#!/usr/bin/env python3
"""Score a trained denoiser on the floor evaluation, exactly as the probe is scored.

    python scripts/eval_denoise.py --checkpoint <out>/best.pt --truth <truth_v2> --tag T --out res.jsonl
        [--near DIR --annot NPZ] [--draws K] [--complete-truth DIR]

Same test events (truth files ``ev260``-``ev387`` of the probe split), same cells,
same windows, same ``Q0`` and the same :func:`helix.probe.resolution.scalars` as
scripts/eval_resolution.py -- only the per-cell prediction comes from the trained
model instead of a probe on frozen features, so the numbers are directly
comparable with every probe result. Cells no token covers are predicted 0.

Rows (one per score type: ``<tag>`` charge, ``<tag>_pres`` presence, ``<tag>_gated``):
  * the STORED corpus (the original rows; every key and value unchanged);
  * with ``--draws K``: the same model on K fresh noise draws of the same events
    (scripts/eval_draws.py: the corpus builder's torch-GPU noise path with seed
    noise_seed + k * 0x9E3779B1, cached on CFS, bit-identical on regeneration; the
    classical chain reads the SAME digitised planes, scripts/eval_classical.py),
    tagged ``<tag>@d<k>``, and an aggregate row ``<tag>@draws<K>``: the mean over draws,
    with intervals combining the event bootstrap of the draw mean and the
    draw-to-draw sd (lo/hi = mean -/+ sqrt(boot_part^2 + draw_sd^2 / K)); the
    aggregate row also carries ``draw_sd`` (the spread of a single draw) and ``per_draw``.
    Single-draw numbers fluctuate by ~0.01 pooled / ~0.025 per plane at 0.1-0.2 MeV
    (baselines/uboone_sp/dense/results/draws.md).
Keys added next to the existing ones (old keys keep their meaning):
  * per-plane 1%-FPR thresholds: ``floor_pp_eff1pct_<E>`` (far noise windows),
    ``near_pp_eff1pct_<E>`` (near-activity noise windows), pooled over planes with each
    plane at its own threshold, plus ``..._<U|V|Y>`` per plane, and
    ``snr_pp_eff1pct_{far,near}_<snr>`` -- the pooled keys mix planes whose score
    scales differ;
  * bright charge on the map rows: ``res68_16k`` / ``med_16k`` (truth >= 16k e-);
  * regions (the 16 x 8-cell windows, all kinds): ``region_sum_ratio_<range>`` (sum pred
    / sum truth), ``region_med_<range>``, ``region_res68_<range>`` for truth window sums
    in 4k-64k and >= 64k e-, ``region_empty_gt1Q0`` (empty windows predicted > 1 Q0);
  * with ``--complete-truth DIR``: ``ct_<key>`` for every charge key above plus
    ``ct_map_r``, ``ct_presence_auc``, against the truth WITHOUT the production's
    per-sample 25 e- threshold (the hits truth drops ~1.6% of the charge, mostly in
    diffusion tails next to bright charge). DIR holds either per-event ``evNNN.npz``
    (scripts/build_eval_complete_truth.py) or the corpus layout
    ``<run>/truth_<shard>.npz`` (build_denoise_truth.py format).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRAW_CACHE = "/global/cfs/cdirs/m5238/users/oalter/helix_runs/eval_draws/coeffs"
PLANES = "UVY"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-events", type=int, default=260, help="probe-split events NOT scored (the probe's training set)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--near", default=None, help="scripts/dump_near_windows.py sidecar dir: near-activity noise windows")
    ap.add_argument("--annot", default=None, help="scripts/noise_vs_hits.py --out npz: per-deposit SNR / isolation")
    ap.add_argument("--q0", type=float, default=None,
                    help="use this Q0 instead of recomputing it from the probe-train truth files (which then are "
                         "not read); it must still equal the model's")
    ap.add_argument("--draws", type=int, default=0, help="also score K fresh noise draws (scripts/eval_draws.py)")
    ap.add_argument("--draw-cache", default=DRAW_CACHE)
    ap.add_argument("--no-stored", action="store_true", help="skip the stored-corpus rows (draws only)")
    ap.add_argument("--complete-truth", default=None, help="complete-truth dir: adds ct_* charge keys")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------------------- truth
def scored_files(a):
    files = sorted(glob.glob(os.path.join(a.truth, "ev*.npz")))
    return files, [(ei, f) for ei, f in enumerate(files) if int(os.path.basename(f)[2:5]) >= a.train_events]


def resolve_q0(a, files):
    if a.q0 is not None:              # given, not recomputed: still checked against the model's
        return float(a.q0)
    qs = np.concatenate([np.load(f, allow_pickle=True)["mq"] for f in files if int(os.path.basename(f)[2:5]) < a.train_events])
    return float(np.median(qs[qs > 0]))


class CompleteTruth:
    """Complete-truth charge per fine-cell key (0 where absent)."""

    def __init__(self, root):
        self.root = root

    def event(self, f, shard, event):
        p = os.path.join(self.root, os.path.basename(f))
        if os.path.exists(p):
            z = np.load(p)
            return z["keys"].astype(np.int64), z["q"].astype(np.float64)
        run = os.path.basename(os.path.dirname(shard))
        tag = os.path.basename(shard).replace("sim_wire_coeff_", "").replace(".h5", "")
        z = np.load(os.path.join(self.root, run, f"truth_{tag}.npz"))
        i = int(np.nonzero(z["event"] == event)[0][0])
        a, b = z["offsets"][i], z["offsets"][i + 1]
        return z["keys"][a:b].astype(np.int64), z["q"][a:b].astype(np.float64)

    @staticmethod
    def lookup(K, Q, keys):
        keys = np.asarray(keys, np.int64)
        j = np.minimum(np.searchsorted(K, keys), max(len(K) - 1, 0))
        return np.where((len(K) > 0) & (K[j] == keys), Q[j], 0.0) if len(K) else np.zeros(len(keys))


# ----------------------------------------------------------------------------------------- prediction
def collect(a, ev_list, Q0, tokens_of, predict_fn, ct=None):
    """Run ``predict_fn(B, bl, keys) -> (out, ok, pres)`` over the scored events, exactly as
    the original loop did; ``tokens_of(z, f) -> (B, bl)``. Returns the arrays ``score`` needs."""
    from helix.probe import resolution as R
    P, PR, Y, E, PL, WQ, WID, WP, WPR, META, QC, WQC = [], [], [], [], [], [], [], [], [], [], [], []
    for ei, f in ev_list:
        z = np.load(f, allow_pickle=True)
        B, bl = tokens_of(z, f)
        if ct is not None:
            CK, CQ = ct.event(f, str(z["shard"]), int(z["event"]))

        def predict(keys):
            return predict_fn(B, bl, keys)

        pm, okm, prm = predict(z["mrows"])
        y = np.log1p(z["mq"] / Q0).astype(np.float32)
        P.append(pm[okm]); PR.append(prm[okm]); Y.append(y[okm]); E.append(np.full(okm.sum(), ei))
        PL.append((R.unkey(z["mrows"][okm])[0] % 3).astype(np.int8))
        if ct is not None:
            QC.append(CompleteTruth.lookup(CK, CQ, z["mrows"][okm]))
        if len(z["wkey"]):
            pw_, _, pr_ = predict(z["wkey"])                  # uncovered cells stay 0
            base = len(META)
            mm = json.loads(str(z["meta"]))
            for m_ in mm:
                m_["ev"] = ei
            META += mm; WQ.append(z["wq"]); WID.append(z["wid"] + base); WP.append(pw_); WPR.append(pr_)
            if ct is not None:
                WQC.append(CompleteTruth.lookup(CK, CQ, z["wkey"]))
        if a.near:
            zn = np.load(os.path.join(a.near, os.path.basename(f)), allow_pickle=True)
            if len(zn["wkey"]):
                pw_, _, pr_ = predict(zn["wkey"])
                base = len(META)
                mm = json.loads(str(zn["meta"]))
                for m_ in mm:
                    m_["ev"] = ei
                META += mm; WQ.append(zn["wq"]); WID.append(zn["wid"] + base); WP.append(pw_); WPR.append(pr_)
                if ct is not None:
                    WQC.append(CompleteTruth.lookup(CK, CQ, zn["wkey"]))
        if ei % 32 == 0:
            print(f"  event {ei}", flush=True)
    return dict(P=P, PR=PR, Y=Y, E=E, PL=PL, WQ=WQ, WID=WID, WP=WP, WPR=WPR, META=META, QC=QC, WQC=WQC)


def context(a, C, Q0, cfg):
    """Annotations + the flat arrays every score shares (the original post-loop code)."""
    META = C["META"]
    if a.annot:                                               # deposit SNR (0: no clean signal) and isolation
        A = np.load(a.annot)
        ann = {k: (float(s) if pk > 0 else 0.0, bool(fo <= 0.25 * q_))
               for k, s, pk, fo, q_ in zip(zip(A["ev"].tolist(), A["g"].tolist(), A["w0"].tolist(), A["t0"].tolist()),
                                           A["snr"], A["peak"], A["foreign"], A["q"])}
        for m_ in META:
            k_ = (m_["ev"], m_["g"], m_["w0"], m_["t0"])
            if m_["kind"] == "iso" and k_ in ann:
                m_["snr"], m_["clean"] = ann[k_]
    y, e, pl = map(np.concatenate, (C["Y"], C["E"], C["PL"]))
    wq, wid = map(np.concatenate, (C["WQ"], C["WID"]))
    starts = np.searchsorted(wid, np.arange(len(META))); ends = np.append(starts[1:], len(wid))
    toff = {i: cfg.toff[i] for i in range(3)}
    q = np.expm1(y) * Q0; q30 = np.quantile(q[q > 0], 0.3)        # as eval_resolution: truth in charge units
    ctx = dict(META=META, y=y, e=e, pl=pl, wq=wq, wid=wid, starts=starts, ends=ends, toff=toff, q=q, q30=q30,
               Q0=Q0, cfg=cfg, qc=None, wqc=None)
    if C["QC"]:
        ctx["qc"] = np.concatenate(C["QC"]); ctx["wqc"] = np.concatenate(C["WQC"])
    return ctx


# ----------------------------------------------------------------------------------------- new keys
def pp_scalars(W, snr_bins=((0, 1e-9), (1e-9, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, 1e9))):
    """1%-FPR efficiencies with a threshold PER PLANE (far: ``bg`` windows; near: ``bgn``),
    pooled over planes (each plane at 1% FPR) and per plane. Needs w['p'] = plane % 3."""
    from helix.probe.resolution import ENERGY_BINS
    out = {}
    names = [f"{ENERGY_BINS[i]:g}-{ENERGY_BINS[i + 1]:g}" for i in range(len(ENERGY_BINS) - 1)]
    for lab, kind in (("floor", "bg"), ("near", "bgn")):
        thr = {}
        for p in range(3):
            s = np.array([w["score"] for w in W if w["kind"] == kind and w["p"] == p])
            if len(s):
                thr[p] = np.quantile(s, 0.99)
        if len(thr) < 3:
            continue
        dec = [(w, w["score"] > thr[w["p"]]) for w in W if w["kind"] == "iso"]
        for i, nm in enumerate(names):
            d = [x for w, x in dec if w["eb"] == i]
            if d:
                out[f"{lab}_pp_eff1pct_{nm}"] = float(np.mean(d))
            for p in range(3):
                d = [x for w, x in dec if w["eb"] == i and w["p"] == p]
                if d:
                    out[f"{lab}_pp_eff1pct_{nm}_{PLANES[p]}"] = float(np.mean(d))
        if any("snr" in w for w, _ in dec):
            for lo, hi in snr_bins:
                d = [x for w, x in dec if w.get("clean") and lo <= w.get("snr", -1) < hi]
                tag = "nosignal" if hi <= 1e-9 else f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}-inf"
                if d:
                    out[f"snr_pp_eff1pct_{'far' if lab == 'floor' else 'near'}_{tag}"] = float(np.mean(d))
    return out


def charge_scalars(qp, q, kz, wsum_p, wsum_t, Q0, prefix=""):
    """Bright-charge and region keys (point values) for predicted charge ``qp`` on map rows
    with truth ``q``; window sums (pred, truth); ``kz``: per (event, plane) Fisher z or None."""
    r = {}
    if kz is not None:
        r[prefix + "map_r"] = float(np.tanh(np.mean(kz))) if len(kz) else float("nan")
    if prefix:
        from helix.probe.resolution import auc
        r[prefix + "haze_frac_0.05Q0"] = float(np.mean(qp[q == 0] > 0.05 * Q0)) if (q == 0).any() else float("nan")
        for lo, hi, nm in ((2000, 16000, "2k_16k"),):
            sel = (q >= lo) & (q < hi)
            rat = qp[sel] / q[sel]
            r[prefix + f"res68_{nm}"] = float((np.quantile(rat, 0.84) - np.quantile(rat, 0.16)) / 2) if sel.any() else float("nan")
            r[prefix + f"med_{nm}"] = float(np.median(rat)) if sel.any() else float("nan")
    sel = q >= 16000
    rat = qp[sel] / q[sel]
    r[prefix + "res68_16k"] = float((np.quantile(rat, 0.84) - np.quantile(rat, 0.16)) / 2) if sel.any() else float("nan")
    r[prefix + "med_16k"] = float(np.median(rat)) if sel.any() else float("nan")
    for lo, hi, nm in ((4e3, 64e3, "4k_64k"), (64e3, np.inf, "64k")):
        m = (wsum_t >= lo) & (wsum_t < hi)
        if m.any():
            rr = wsum_p[m] / wsum_t[m]
            r[prefix + f"region_sum_ratio_{nm}"] = float(wsum_p[m].sum() / wsum_t[m].sum())
            r[prefix + f"region_med_{nm}"] = float(np.median(rr))
            r[prefix + f"region_res68_{nm}"] = float((np.quantile(rr, 0.84) - np.quantile(rr, 0.16)) / 2)
    m0 = wsum_t <= 0
    r[prefix + "region_empty_gt1Q0"] = float(np.mean(wsum_p[m0] > Q0)) if m0.any() else float("nan")
    return r


# ----------------------------------------------------------------------------------------- scoring
def score(a, ctx, tag, p, wscore, header, charge=True, write=True):
    """Every metric for one per-cell score ``p`` (map rows) and window cell score
    ``wscore``: charge (expm1(max(p,0))*Q0, exactly eval_resolution's window input) or
    presence probability. Returns (row, point values, bootstrap dicts)."""
    from helix.probe import resolution as R
    META, y, e, pl, wq, starts, ends, toff, q, q30, Q0, cfg = (ctx[k] for k in (
        "META", "y", "e", "pl", "wq", "starts", "ends", "toff", "q", "q30", "Q0", "cfg"))
    W = R.window_stats(wscore, wq, META, starts, ends, cfg.pw, cfg.pt, cfg.delta[0], toff)
    for w_, m_ in zip(W, META):
        w_.update({k_: m_[k_] for k_ in ("snr", "clean") if k_ in m_})
        w_["p"] = m_["g"] % 3                                   # plane, for the per-plane thresholds
    kz = {}
    for k in np.unique(e * 10 + pl):
        m = (e * 10 + pl) == k
        if m.sum() > 20 and y[m].std() > 0 and p[m].std() > 0:
            kz[int(k)] = float(np.arctanh(np.clip(np.corrcoef(p[m], y[m])[0, 1], -0.999, 0.999)))
    evs = np.unique(e)
    rows_of = {v: np.nonzero(e == v)[0] for v in evs}
    wins_of = {v: [w for w in W if w["ev"] == v] for v in evs}
    z_of = {v: [z_ for k, z_ in kz.items() if k // 10 == v] for v in evs}
    point = R.scalars(p, q, list(kz.values()), W, q30)
    point.update(R.near_scalars(W))
    point.update(pp_scalars(W))
    # charge keys on the predicted charge (charge-type scores only)
    is_charge = charge and not (p.min() >= 0 and p.max() <= 1 and tag.endswith("_pres"))
    if is_charge:
        qp_all = np.expm1(np.maximum(p, 0)) * Q0
        wsum_p = np.add.reduceat(wscore, starts); wsum_p[ends == starts] = 0
        wsum_t = np.add.reduceat(wq, starts); wsum_t[ends == starts] = 0
        win_ev = np.array([m_["ev"] for m_ in META])
        win_of = {v: np.nonzero(win_ev == v)[0] for v in evs}
        qc, wqc = ctx.get("qc"), ctx.get("wqc")
        kzc = {}
        if qc is not None:
            yc = np.log1p(qc / Q0)
            for k in np.unique(e * 10 + pl):
                m = (e * 10 + pl) == k
                if m.sum() > 20 and yc[m].std() > 0 and p[m].std() > 0:
                    kzc[int(k)] = float(np.arctanh(np.clip(np.corrcoef(p[m], yc[m])[0, 1], -0.999, 0.999)))
            wsum_c = np.add.reduceat(wqc, starts); wsum_c[ends == starts] = 0
            zc_of = {v: [z_ for k, z_ in kzc.items() if k // 10 == v] for v in evs}

        def cscal(idx, widx, zl, zcl):
            out = charge_scalars(qp_all[idx], q[idx], None, wsum_p[widx], wsum_t[widx], Q0)
            if qc is not None:
                out.update(charge_scalars(qp_all[idx], qc[idx], zcl, wsum_p[widx], wsum_c[widx], Q0, prefix="ct_"))
                out["ct_presence_auc"] = R.auc(p[idx][qc[idx] > 0], p[idx][qc[idx] == 0])
            return out
        point.update(cscal(np.arange(len(p)), np.arange(len(META)), None, list(kzc.values())))
    rng = np.random.default_rng(0); boots = []
    for _ in range(a.boot):
        sm = rng.choice(evs, len(evs), replace=True)
        idx = np.concatenate([rows_of[v] for v in sm])
        wb_ = [w for v in sm for w in wins_of[v]]
        b = dict(R.scalars(p[idx], q[idx], [z_ for v in sm for z_ in z_of[v]], wb_, q30), **R.near_scalars(wb_))
        b.update(pp_scalars(wb_))
        if is_charge:
            widx = np.concatenate([win_of[v] for v in sm])
            b.update(cscal(idx, widx, None, [z_ for v in sm for z_ in zc_of[v]] if qc is not None else None))
        boots.append(b)
    r = {"val_mse": float(np.mean((np.concatenate(header["P"]) - y) ** 2))}
    if not charge or (p.min() >= 0 and p.max() <= 1 and tag.endswith("_pres")):
        pass                                                   # a probability / detection statistic: no charge scalars
    else:                                                      # quick charge scalars on the map rows
        qp = np.expm1(np.maximum(p, 0)) * Q0
        r["haze_frac_0.05Q0"] = float(np.mean(qp[q == 0] > 0.05 * Q0))
        sel = (q >= 2000) & (q < 16000)
        rat = qp[sel] / q[sel]
        r["res68_2k_16k"] = float((np.quantile(rat, 0.84) - np.quantile(rat, 0.16)) / 2)
        r["med_2k_16k"] = float(np.median(rat))
    for k, v in point.items():
        bv = np.array([b_.get(k, np.nan) for b_ in boots], float)
        lo, hi = np.nanpercentile(bv, [16, 84]) if np.isfinite(bv).any() else (np.nan, np.nan)
        r[k] = [round(v, 4), round(float(lo), 4), round(float(hi), 4)]
    res = dict(tag=tag, checkpoint=header["checkpoint"], step=header["step"], Q0=Q0, n_windows=len(META),
               train_events=header["train_events"], init=header["init"])
    res.update(header.get("extra", {}))
    res["trained"] = r
    if write:
        with open(a.out.replace(".jsonl", f"_{tag}_trained_windows.json"), "w") as fh:
            json.dump([dict({k: w[k] for k in ("kind", "score", "eb", "dip", "edge", "sep", "snr", "clean") if k in w},
                            ev=m_.get("ev"), g=m_.get("g"), w0=m_.get("w0"), t0=m_.get("t0"))
                       for w, m_ in zip(W, META)], fh, default=float)       # where each window is, for diagnosis
        with open(a.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(tag, json.dumps({k: r[k] for k in ("map_r", "faint_auc", "floor_eff1pct_0.1-0.2", "floor_auc_0-0.1")
                               if k in r}), flush=True)
    raw = {"val_mse": r["val_mse"], **{k: r[k] for k in ("haze_frac_0.05Q0", "res68_2k_16k", "med_2k_16k") if k in r}}
    return res, dict(point, **raw), boots


def aggregate(a, tag, per_draw, header):
    """Aggregate row over K draws: mean, intervals = event bootstrap of the draw mean
    (same event resamples in every draw) combined with the draw-to-draw sd / sqrt(K)."""
    K = len(per_draw)
    keys = list(per_draw[0][0].keys())
    r, dsd, pdv = {}, {}, {}
    for k in keys:
        vals = np.array([pd[0].get(k, np.nan) for pd in per_draw], float)
        m = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1)) if K > 1 else float("nan")
        dsd[k] = round(sd, 5); pdv[k] = [round(float(x), 5) for x in vals]
        if per_draw[0][1] and k in per_draw[0][1][0]:
            bm = np.nanmean(np.array([[b.get(k, np.nan) for b in pd[1]] for pd in per_draw], float), 0)
            if np.isfinite(bm).any():
                p16, p84 = np.nanpercentile(bm, [16, 84])
                bmean = float(np.nanmean(bm))
                lo = m - np.sqrt((bmean - p16) ** 2 + (sd ** 2 / K if K > 1 else 0.0))
                hi = m + np.sqrt((p84 - bmean) ** 2 + (sd ** 2 / K if K > 1 else 0.0))
                r[k] = [round(m, 4), round(float(lo), 4), round(float(hi), 4)]
                continue
        r[k] = round(m, 6)
    res = dict(tag=tag, checkpoint=header["checkpoint"], step=header["step"], Q0=header["Q0"], n_windows=header["n_windows"],
               train_events=header["train_events"], init=header["init"], source=f"draws{K}", n_draws=K,
               trained=r, draw_sd=dsd, per_draw=pdv)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(tag, json.dumps({k: r[k] for k in ("near_pp_eff1pct_0.1-0.2", "near_eff1pct_0.1-0.2") if k in r}), flush=True)
    return res


def score_all(a, ctx, C, header, tag, presence):
    """The three score types of one source (charge, presence, gated), as the original main."""
    Q0 = ctx["Q0"]
    out = []
    wp = np.concatenate(C["WP"])
    out.append((tag, *score(a, ctx, tag, np.concatenate(C["P"]), np.expm1(np.maximum(wp, 0)) * Q0, header)[1:]))
    if presence:                      # the same cells scored by "is there charge here"
        out.append((tag + "_pres", *score(a, ctx, tag + "_pres", np.concatenate(C["PR"]), np.concatenate(C["WPR"]), header)[1:]))
        # the hurdle's output map: charge only where presence > 0.5
        gp = np.where(np.concatenate(C["PR"]) > 0.5, np.concatenate(C["P"]), 0.0)
        gw = np.where(np.concatenate(C["WPR"]) > 0.5, wp, 0.0)
        out.append((tag + "_gated", *score(a, ctx, tag + "_gated", gp, np.expm1(np.maximum(gw, 0)) * Q0, header)[1:]))
    return out


def main():
    a = parse_args()
    import torch
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.probe import resolution as R

    dev = torch.device("cuda")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    hk = meta.get("head_kw") or {}
    presence = bool(hk.get("presence"))
    decoder = hk.get("kind") == "decoder"
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=hk)
    model.load_state_dict(ck["model"]); model.to(dev).eval()
    q0 = float(meta["q0"])
    ds = DenoiseEvents("", [], "", items=[])                  # tokenizer + cfg only
    cfg = ds.cfg

    files, ev_list = scored_files(a)
    Q0 = resolve_q0(a, files)
    if abs(Q0 - q0) > 1e-6 * Q0:
        raise SystemExit(f"model trained with Q0={q0}, this truth has Q0={Q0}: predictions are on another scale")
    ct = CompleteTruth(a.complete_truth) if a.complete_truth else None

    def predict_fn(B, bl, keys):
        if predict_fn.cache[0] is not B:                     # tokens moved to the GPU once per event
            Bt = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in B.items()}
            Bt["n_cells"] = Bt["plane_id"].shape[0]
            predict_fn.cache = (B, Bt)
        Bt = predict_fn.cache[1]
        idx, aux = R.cell_inputs(keys, B["cell_key"], bl, cfg)
        ok = (idx >= 0).any(1)
        out = np.zeros(len(keys), np.float32)
        pres = np.zeros(len(keys), np.float32)
        if ok.any():
            nbr = torch.as_tensor(R.cell_neighbors(keys[ok], B["cell_key"], bl, cfg)).to(dev) if decoder else None
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p = model(Bt, torch.as_tensor(idx[ok]).to(dev), torch.as_tensor(aux[ok]).to(dev), nbr)
            if presence:
                pres[ok] = torch.sigmoid(p[1].float()).cpu().numpy(); p = p[0]
            out[ok] = p.float().cpu().numpy()
        return out, ok, pres
    predict_fn.cache = (None, None)

    def stored_tokens(z, f):
        import h5py
        shard, event = str(z["shard"]), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        _, B, bl = ds.tokens(shard, pos)
        return B, bl

    header = dict(checkpoint=a.checkpoint, step=ck.get("step"), train_events=meta["args"].get("n_events"),
                  init=meta["args"].get("init"))
    if not a.no_stored:
        C = collect(a, ev_list, Q0, stored_tokens, predict_fn, ct)
        ctx = context(a, C, Q0, cfg)
        score_all(a, ctx, C, dict(header, P=C["P"]), a.tag, presence)
    if a.draws > 0:
        import eval_draws as ED
        per = {}
        for k in range(a.draws):
            def draw_tokens(z, f, k=k):
                shard = str(z["shard"])
                path = ED.build(shard, int(z["event"]), k, a.draw_cache, ev_tag=int(os.path.basename(f)[2:5]))
                _, B, bl = ED.tokens(path, shard, ds)
                return B, bl
            C = collect(a, ev_list, Q0, draw_tokens, predict_fn, ct)
            ctx = context(a, C, Q0, cfg)
            hdr = dict(header, P=C["P"], extra=dict(source=f"d{k}", draw=k))
            for t, point, boots in score_all(a, ctx, C, hdr, f"{a.tag}@d{k}", presence):
                per.setdefault(t.replace(f"@d{k}", ""), []).append((point, boots))
        for t, pd in per.items():
            base = t[len(a.tag):]
            aggregate(a, f"{a.tag}@draws{a.draws}{base}", pd,
                      dict(header, Q0=Q0, n_windows=len(ctx["META"])))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
