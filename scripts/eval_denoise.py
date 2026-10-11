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
With ``--products PARAMS.json`` (scripts/denoise_products.py fit; needs --complete-truth and a
presence head) the model is also run on EVERY token-covered readout cell of each event, the
products are formed (prior-corrected presence p_cal, calorimetry map E[q|data], clean map =
calo where p_cal > tau) and three more rows are written per source: ``<tag>_cal`` (p_cal as the
cell / window score), ``<tag>_calo`` and ``<tag>_clean`` (charge maps, every charge key). Every
row of the run (old ones included) also gains, bootstrapped over the same event resamples:
  * ``floor_iso_pp_eff1pct_<E>[_<P>]`` / ``near_iso_pp_eff1pct_<E>[_<P>]``: the per-plane 1%-FPR
    efficiencies (far / near thresholds) on ISOLATED positives only -- iso windows whose bg padding
    (16 wires / 128 ticks each side, window excluded) holds no complete-truth cell > 25 e-;
  * charge maps (<tag>, _gated, _calo, _clean), over the full readout vs the complete truth:
    ``lin_cell_<bin>`` / ``lin_region_<bin>`` = sum truth / sum prediction per bin of the
    PREDICTION (cells; 16 x 8-cell regions), ``tot_ratio[_<P>]`` = sum pred / sum truth,
    ``fm_false_frac`` (predicted charge on cells with truth <= 25 e-, / total truth),
    ``fm_zeroed_frac`` (truth on cells predicted 0, / total), ``fm_l1_frac`` (sum |pred - truth|
    / total), ``fm_haze_frac_0.05Q0`` (covered cells with truth <= 25 e- predicted > 0.05 Q0);
  * presence maps (_pres raw, _cal calibrated), all covered cells: ``cal_ece``, ``cal_brier``,
    ``cal_logloss`` [``_<P>``] against complete q > c, c = the checkpoint's support_min_q
    (default 0), and ``cal_own_*`` against the model's training truth when that is not the
    complete truth (the label the prior correction was fitted for).
Without --products every row and key is unchanged.
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
    ap.add_argument("--products", default=None,
                    help="scripts/denoise_products.py fit json: adds the _cal/_calo/_clean rows and the full-map keys "
                         "(needs --complete-truth and a presence head)")
    ap.add_argument("--products-key", default=None, help="entry of --products to use (default: --tag)")
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
def collect(a, ev_list, Q0, tokens_of, predict_fn, ct=None, prod=None):
    """Run ``predict_fn(B, bl, keys) -> (out, ok, pres)`` over the scored events, exactly as
    the original loop did; ``tokens_of(z, f) -> (B, bl)``. Returns the arrays ``score`` needs.
    With ``prod`` (--products) also the full-map products per event (``tokens_of.ce``: the
    event's coefficients, for the token footprints)."""
    from helix.probe import resolution as R
    P, PR, Y, E, PL, WQ, WID, WP, WPR, META, QC, WQC = [], [], [], [], [], [], [], [], [], [], [], []
    FM, RP, WRP = {}, [], []
    for ei, f in ev_list:
        z = np.load(f, allow_pickle=True)
        B, bl = tokens_of(z, f)
        if ct is not None:
            CK, CQ = ct.event(f, str(z["shard"]), int(z["event"]))
        if prod is not None:
            zn_ = np.load(os.path.join(a.near, os.path.basename(f)), allow_pickle=True) if a.near else None
            F = prod["event"](tokens_of.ce, B, bl, z, zn_, CK, CQ, f)
            FM[ei] = F["stats"]

        def predict(keys):
            return predict_fn(B, bl, keys)

        pm, okm, prm = predict(z["mrows"])
        y = np.log1p(z["mq"] / Q0).astype(np.float32)
        P.append(pm[okm]); PR.append(prm[okm]); Y.append(y[okm]); E.append(np.full(okm.sum(), ei))
        PL.append((R.unkey(z["mrows"][okm])[0] % 3).astype(np.int8))
        if ct is not None:
            QC.append(CompleteTruth.lookup(CK, CQ, z["mrows"][okm]))
        if prod is not None:
            RP.append(F["at"](z["mrows"][okm]))
        if len(z["wkey"]):
            pw_, _, pr_ = predict(z["wkey"])                  # uncovered cells stay 0
            base = len(META)
            mm = json.loads(str(z["meta"]))
            for m_ in mm:
                m_["ev"] = ei
                if prod is not None and m_["kind"] == "iso":
                    m_["isolated"] = F["ring_max"](m_) <= ISO_RING_MAX
            META += mm; WQ.append(z["wq"]); WID.append(z["wid"] + base); WP.append(pw_); WPR.append(pr_)
            if ct is not None:
                WQC.append(CompleteTruth.lookup(CK, CQ, z["wkey"]))
            if prod is not None:
                WRP.append(F["at"](z["wkey"]))
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
                if prod is not None:
                    WRP.append(F["at"](zn["wkey"]))
        if ei % 32 == 0:
            print(f"  event {ei}", flush=True)
    out = dict(P=P, PR=PR, Y=Y, E=E, PL=PL, WQ=WQ, WID=WID, WP=WP, WPR=WPR, META=META, QC=QC, WQC=WQC)
    if prod is not None:
        out.update(FM=FM, RP={k: np.concatenate([d[k] for d in RP]) for k in RP[0]},
                   WRP={k: np.concatenate([d[k] for d in WRP]) for k in WRP[0]})
    return out


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
    if "FM" in C:
        ctx["products"] = True
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
def score(a, ctx, tag, p, wscore, header, charge=True, write=True, fm=None):
    """Every metric for one per-cell score ``p`` (map rows) and window cell score
    ``wscore``: charge (expm1(max(p,0))*Q0, exactly eval_resolution's window input) or
    presence probability. ``fm(events) -> keys``: full-map keys from per-event sums (with
    --products), bootstrapped with the same event resamples. Returns (row, point values,
    bootstrap dicts)."""
    from helix.probe import resolution as R
    META, y, e, pl, wq, starts, ends, toff, q, q30, Q0, cfg = (ctx[k] for k in (
        "META", "y", "e", "pl", "wq", "starts", "ends", "toff", "q", "q30", "Q0", "cfg"))
    W = R.window_stats(wscore, wq, META, starts, ends, cfg.pw, cfg.pt, cfg.delta[0], toff)
    for w_, m_ in zip(W, META):
        w_.update({k_: m_[k_] for k_ in ("snr", "clean", "isolated") if k_ in m_})
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
    prod = ctx.get("products", False)
    if prod:
        point.update(iso_pp_scalars(W))
    if fm is not None:
        point.update(fm(evs))
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
        if prod:
            b.update(iso_pp_scalars(wb_))
        if fm is not None:
            b.update(fm(sm))
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


# ----------------------------------------------------------------------------------------- products (--products)
ISO_RING_MAX = 25.0          # isolated positive: no complete-truth cell > 25 e- in the bg padding ring
QB = np.array([-1, 0, 25, 102, 500, 1e3, 2e3, 4e3, 8e3, 16e3, 64e3, np.inf])
QBL = ["0", "0-25", "25-102", "102-500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "8k-16k", "16k-64k", "64k"]
RB = np.array([-1, 0, 1e3, 4e3, 16e3, 64e3, 256e3, 1e6, np.inf])
RBL = ["0", "0-1k", "1k-4k", "4k-16k", "16k-64k", "64k-256k", "256k-1M", "1M"]
PB = np.array([0, 1e-4, 1e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1 + 1e-9])
REG = (16, 8)                # region = one window footprint, 16 x 8 fine cells
EMPTY_Q = 25.0               # 'empty' for haze / false charge: complete truth <= 25 e-
CHARGE_MAPS = ("", "_gated", "_calo", "_clean")
PROB_MAPS = ("_pres", "_cal")


def iso_pp_scalars(W):
    """pp_scalars with the iso positives restricted to ISOLATED ones (padding ring free of
    complete-truth charge > 25 e-); thresholds from the same bg / bgn windows."""
    Wf = [w for w in W if w["kind"] != "iso" or w.get("isolated")]
    return {k.replace("floor_pp_", "floor_iso_pp_").replace("near_pp_", "near_iso_pp_"): v
            for k, v in pp_scalars(Wf, snr_bins=()).items() if not k.startswith("snr")}


def covered_cells_fn(cfg):
    """covered_cells(ce): every fine cell inside a token's footprint (scripts/plot_denoise.covered_cells)."""
    from helix.model.tokenize import tick_of_tau
    from helix.probe import resolution as R

    def covered_cells(ce):
        k = ce.band < cfg.n_bands
        out = []
        for b in range(cfg.n_bands):
            m = k & (ce.band == b)
            if not m.any():
                continue
            gb, wb, tb = np.unique(np.stack([ce.plane_gid[m], ce.wire[m] // cfg.pw, ce.tau[m] // cfg.pt], 1), axis=0).T
            t_raw = tick_of_tau(tb * cfg.pt, gb, np.full(len(gb), b), cfg)
            f_lo = np.floor(np.maximum(t_raw, 0) / R.FT).astype(np.int64)
            f_hi = np.floor(np.maximum(t_raw + cfg.pt * (1 << cfg.lev[b]), 0) / R.FT).astype(np.int64)
            fw = (wb * cfg.pw // R.FW)[:, None] + np.arange(cfg.pw // R.FW)
            ft = f_lo[:, None] + np.arange(int((f_hi - f_lo).max()) + 1)
            shape = (len(gb), fw.shape[1], ft.shape[1])
            sel = np.broadcast_to((ft <= f_hi[:, None])[:, None, :], shape)
            out.append(R.fkey(np.broadcast_to(gb[:, None, None], shape)[sel],
                              np.broadcast_to(fw[:, :, None], shape)[sel],
                              np.broadcast_to(ft[:, None, :], shape)[sel]))
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)
    return covered_cells


def _plane_maps(keys, vals, shapes):
    from helix.probe.resolution import unkey
    g, w, t = unkey(np.asarray(keys, np.int64))
    out = {}
    for gg, sh in shapes.items():
        M = np.zeros(sh)
        m = (g == gg) & (w >= 0) & (w < sh[0]) & (t >= 0) & (t < sh[1])
        np.add.at(M, (w[m], t[m]), vals[m])
        out[gg] = M
    return out


def _charge_stats(M, T, cov, Q0):
    """Per-event sums for one charge map over the readout: cells binned by PREDICTION
    (n, sum pred, sum truth), 16x8 regions binned by predicted sum, and plane totals."""
    LC = np.zeros((3, len(QB) - 1, 3)); LR = np.zeros((3, len(RB) - 1, 3)); TOT = np.zeros((3, 8))
    for g in M:
        p_, X, Tg, cv = g % 3, M[g], T[g], cov[g]
        b = np.clip(np.searchsorted(QB, X[cv], side="left") - 1, 0, len(QB) - 2)
        for j, w in enumerate((None, X[cv], Tg[cv])):
            LC[p_, :, j] += np.bincount(b, weights=w, minlength=len(QB) - 1)
        sh = X.shape
        H, Wd = -(-sh[0] // REG[0]) * REG[0], -(-sh[1] // REG[1]) * REG[1]
        rs = lambda A: np.pad(A, ((0, H - sh[0]), (0, Wd - sh[1]))).reshape(H // REG[0], REG[0], Wd // REG[1], REG[1]).sum((1, 3))
        Xr, Tr, Cr = rs(X), rs(Tg), rs(cv.astype(float)) > 0
        b = np.clip(np.searchsorted(RB, Xr[Cr], side="left") - 1, 0, len(RB) - 2)
        for j, w in enumerate((None, Xr[Cr], Tr[Cr])):
            LR[p_, :, j] += np.bincount(b, weights=w, minlength=len(RB) - 1)
        emp = Tg <= EMPTY_Q
        TOT[p_] += [X.sum(), Tg.sum(), X[emp].sum(), Tg[X <= 0].sum(), np.abs(X - Tg).sum(),
                    (emp & cv).sum(), (emp & cv & (X > 0.05 * Q0)).sum(), Tg[cv].sum()]
    return dict(LC=LC, LR=LR, TOT=TOT)


def _prob_stats(Pm, lab, cov):
    """Per-event sums for a presence map on covered cells vs a label map: reliability bins
    (n, sum p, sum y) and (n, Brier, log-loss) per plane."""
    REL = np.zeros((3, len(PB) - 1, 3)); BR = np.zeros((3, 3))
    for g in Pm:
        p_, cv = g % 3, cov[g]
        pc, y = Pm[g][cv], lab[g][cv].astype(float)
        b = np.clip(np.searchsorted(PB, pc, side="right") - 1, 0, len(PB) - 2)
        for j, w in enumerate((None, pc, y)):
            REL[p_, :, j] += np.bincount(b, weights=w, minlength=len(PB) - 1)
        pcc = np.clip(pc, 1e-7, 1 - 1e-7)
        BR[p_] += [len(pc), ((pc - y) ** 2).sum(), -(y * np.log(pcc) + (1 - y) * np.log1p(-pcc)).sum()]
    return dict(REL=REL, BR=BR)


def fm_keys(S, kind):
    """Full-map keys from per-event sums ``S`` (summed over the chosen events)."""
    r = {}
    div = lambda x, y: float(x / y) if y > 0 else float("nan")       # noqa: E731
    if kind == "charge":
        LC, LR, TOT = S["LC"].sum(0), S["LR"].sum(0), S["TOT"]
        for i in range(1, len(QBL)):                                 # truth / pred per predicted-charge bin
            r[f"lin_cell_{QBL[i]}"] = div(LC[i, 2], LC[i, 1])
        for i in range(1, len(RBL)):
            r[f"lin_region_{RBL[i]}"] = div(LR[i, 2], LR[i, 1])
        T = TOT.sum(0)
        r["tot_ratio"] = div(T[0], T[1])
        for p in range(3):
            r[f"tot_ratio_{PLANES[p]}"] = div(TOT[p, 0], TOT[p, 1])
        r["fm_false_frac"] = div(T[2], T[1]); r["fm_zeroed_frac"] = div(T[3], T[1])
        r["fm_l1_frac"] = div(T[4], T[1]); r["fm_haze_frac_0.05Q0"] = div(T[6], T[5])
        return r
    for pre, rk, bk in (("cal_", "REL", "BR"), ("cal_own_", "RELo", "BRo")):
        if rk not in S:
            continue
        REL, BR = S[rk], S[bk]
        for p, nm in [(slice(None), "")] + [(p, f"_{PLANES[p]}") for p in range(3)]:
            R_ = REL[p].reshape(-1, len(PB) - 1, 3).sum(0); B_ = BR[p].reshape(-1, 3).sum(0)
            r[f"{pre}ece{nm}"] = div(np.abs(R_[:, 1] - R_[:, 2]).sum(), R_[:, 0].sum())
            r[f"{pre}brier{nm}"] = div(B_[1], B_[0]); r[f"{pre}logloss{nm}"] = div(B_[2], B_[0])
    return r


def fm_fn(C, name):
    """fm(events) for the score of map ``name`` (row suffix), or None without --products."""
    if "FM" not in C:
        return None
    kind = "charge" if name in CHARGE_MAPS else "prob"
    FM = C["FM"]

    def fm(evs):
        S = {}
        for v in evs:
            for k, x in FM[v][name].items():
                S[k] = S[k] + x if k in S else x.copy()
        return fm_keys(S, kind)
    return fm


class Products:
    """Full-map prediction of one event and its products (scripts/denoise_products.py)."""

    def __init__(self, prm, cfg, Q0, predict_full, support_min_q, own=None):
        import denoise_products as DP
        from helix.tpc.geometry import load_plane_registry
        self.DP, self.prm, self.Q0, self.predict_full, self.c, self.own = DP, prm, Q0, predict_full, support_min_q, own
        reg = load_plane_registry("cubic_wireplane_geometry.json")
        self.shapes = {g: (-(-reg[g]["n_wires"] // 2), -(-reg[g]["n_ticks"] // 16)) for g in range(6)}
        self.covered = covered_cells_fn(cfg)

    def event(self, ce, B, bl, z, zn, CK, CQ, f):
        from helix.probe import resolution as R
        Q0, sh = self.Q0, self.shapes
        o = np.argsort(CK); CK, CQ = CK[o], CQ[o]
        parts = [self.covered(ce), z["mrows"], z["wkey"]] + ([zn["wkey"]] if zn is not None else [])
        keys = np.unique(np.concatenate(parts).astype(np.int64))
        g, fw, ft = R.unkey(keys)
        inside = np.zeros(len(keys), bool)
        for gg, s_ in sh.items():
            inside |= (g == gg) & (fw >= 0) & (fw < s_[0]) & (ft >= 0) & (ft < s_[1])
        keys = keys[inside]
        y, ok, pr = self.predict_full(B, bl, keys)
        q = np.where(ok, np.expm1(np.maximum(y, 0)) * Q0, 0.0)
        Qm, Pm = _plane_maps(keys, q, sh), _plane_maps(keys, np.where(ok, pr, 0.0), sh)
        Cm = {k: v > 0 for k, v in _plane_maps(keys, ok.astype(float), sh).items()}
        Tc = _plane_maps(CK, CQ, sh)
        prods = {gg: self.DP.products_maps(Qm[gg], Pm[gg], Cm[gg], gg % 3, self.prm) for gg in sh}
        maps = {"": Qm, "_gated": {gg: Qm[gg] * (Pm[gg] > 0.5) for gg in sh},
                "_calo": {gg: prods[gg]["calo"] for gg in sh}, "_clean": {gg: prods[gg]["clean"] for gg in sh}}
        stats = {k: _charge_stats(v, Tc, Cm, Q0) for k, v in maps.items()}
        lab = {gg: Tc[gg] > self.c for gg in sh}
        if self.own is not None:
            OK, OQ = self.own.event(f, str(z["shard"]), int(z["event"]))
            To = _plane_maps(OK, OQ, sh)
            labo = {gg: To[gg] > self.c for gg in sh}
        for k, Pmap in (("_pres", Pm), ("_cal", {gg: prods[gg]["p_cal"] for gg in sh})):
            stats[k] = _prob_stats(Pmap, lab, Cm)
            if self.own is not None:
                so = _prob_stats(Pmap, labo, Cm)
                stats[k].update(RELo=so["REL"], BRo=so["BR"])

        def at(qk):
            gq, wq, tq = R.unkey(np.asarray(qk, np.int64))
            out = {k: np.zeros(len(gq), np.float32) for k in ("p_cal", "calo", "clean")}
            for gg, s_ in sh.items():
                m = (gq == gg) & (wq >= 0) & (wq < s_[0]) & (tq >= 0) & (tq < s_[1])
                for k in out:
                    out[k][m] = prods[gg][k][wq[m], tq[m]]
            return out

        def ring_max(m_):
            """Max complete-truth cell in the bg padding box (16 wires / 128 ticks each side), window excluded."""
            gg, w0, t0 = int(m_["g"]), int(m_["w0"]), int(m_["t0"])
            fw_ = np.arange((w0 - 16) // 2, (w0 + 48) // 2); ft_ = np.arange((t0 - 128) // 16, (t0 + 256) // 16)
            FWm, FTm = np.meshgrid(fw_, ft_, indexing="ij")
            insd = (FWm >= w0 // 2) & (FWm < w0 // 2 + 16) & (FTm >= t0 // 16) & (FTm < t0 // 16 + 8)
            okk = (FWm >= 0) & ~insd
            qv = CompleteTruth.lookup(CK, CQ, R.fkey(np.full(int(okk.sum()), gg), FWm[okk], FTm[okk]))
            return float(qv.max()) if len(qv) else 0.0

        return dict(stats=stats, at=at, ring_max=ring_max)


def score_all(a, ctx, C, header, tag, presence):
    """The three score types of one source (charge, presence, gated), as the original main."""
    Q0 = ctx["Q0"]
    out = []
    wp = np.concatenate(C["WP"])
    out.append((tag, *score(a, ctx, tag, np.concatenate(C["P"]), np.expm1(np.maximum(wp, 0)) * Q0, header,
                            fm=fm_fn(C, ""))[1:]))
    if presence:                      # the same cells scored by "is there charge here"
        out.append((tag + "_pres", *score(a, ctx, tag + "_pres", np.concatenate(C["PR"]), np.concatenate(C["WPR"]), header,
                                          fm=fm_fn(C, "_pres"))[1:]))
        # the hurdle's output map: charge only where presence > 0.5
        gp = np.where(np.concatenate(C["PR"]) > 0.5, np.concatenate(C["P"]), 0.0)
        gw = np.where(np.concatenate(C["WPR"]) > 0.5, wp, 0.0)
        out.append((tag + "_gated", *score(a, ctx, tag + "_gated", gp, np.expm1(np.maximum(gw, 0)) * Q0, header,
                                           fm=fm_fn(C, "_gated"))[1:]))
    if "RP" in C:                     # --products: calibrated presence, calorimetry map, clean map
        RP, WRP = C["RP"], C["WRP"]
        out.append((tag + "_cal", *score(a, ctx, tag + "_cal", RP["p_cal"], WRP["p_cal"], header, charge=False,
                                         fm=fm_fn(C, "_cal"))[1:]))
        for nm in ("calo", "clean"):
            out.append((tag + f"_{nm}", *score(a, ctx, tag + f"_{nm}", np.log1p(RP[nm] / Q0), WRP[nm], header,
                                               fm=fm_fn(C, f"_{nm}"))[1:]))
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
    if a.products and (ct is None or not presence):
        raise SystemExit("--products needs --complete-truth and a presence head")

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

    def predict_full(B, bl, keys, chunk=65536):
        """Every given cell, encoder run once and the head in chunks (dense/model_dump2.py)."""
        if predict_fn.cache[0] is not B:                     # tokens to the GPU, shared with predict_fn
            Bt = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in B.items()}
            Bt["n_cells"] = Bt["plane_id"].shape[0]
            predict_fn.cache = (B, Bt)
        Bt = predict_fn.cache[1]
        idx, aux = R.cell_inputs(keys, B["cell_key"], bl, cfg)
        ok = (idx >= 0).any(1)
        out = np.zeros(len(keys), np.float32); pres = np.zeros(len(keys), np.float32)
        sel_all = np.nonzero(ok)[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.fm.encode(Bt)
            for s0 in range(0, len(sel_all), chunk):
                sel = sel_all[s0:s0 + chunk]
                ti, ta = torch.as_tensor(idx[sel]).to(dev), torch.as_tensor(aux[sel]).to(dev)
                if decoder:
                    nbr = torch.as_tensor(R.cell_neighbors(keys[sel], B["cell_key"], bl, cfg)).to(dev)
                    o = model.head(feats, ti, ta, nbr=nbr, B=Bt)
                else:
                    o = model.head(feats, ti, ta)
                if presence:
                    pres[sel] = torch.sigmoid(o[1].float()).cpu().numpy(); o = o[0]
                out[sel] = o.float().cpu().numpy()
        return out, ok, pres

    prod = None
    if a.products:
        prm = json.load(open(a.products))[a.products_key or a.tag]
        c = float(meta["args"].get("support_min_q") or 0.0)
        hurdle = bool(meta.get("cond_charge") or meta["args"].get("cond_charge"))
        if abs(prm["support_min_q"] - c) > 1e-9 or prm["hurdle"] != hurdle or abs(prm["q0"] - Q0) > 1e-6 * Q0:
            raise SystemExit(f"--products entry does not match this checkpoint (support {prm['support_min_q']} vs {c}, "
                             f"hurdle {prm['hurdle']} vs {hurdle})")
        troot = prm["fit"]["truth_root"]
        own = None if os.path.realpath(troot) == os.path.realpath(a.complete_truth) else CompleteTruth(troot)
        P_ = Products(prm, cfg, Q0, predict_full, c, own)
        prod = dict(event=P_.event)

    def stored_tokens(z, f):
        import h5py
        shard, event = str(z["shard"]), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        ce, B, bl = ds.tokens(shard, pos)
        stored_tokens.ce = ce
        return B, bl

    header = dict(checkpoint=a.checkpoint, step=ck.get("step"), train_events=meta["args"].get("n_events"),
                  init=meta["args"].get("init"))
    if not a.no_stored:
        C = collect(a, ev_list, Q0, stored_tokens, predict_fn, ct, prod)
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
                if prod is not None:                         # token footprints of the draw
                    from types import SimpleNamespace
                    zc = np.load(path)
                    draw_tokens.ce = SimpleNamespace(**{k: zc[k] for k in ("band", "plane_gid", "wire", "tau")})
                return B, bl
            C = collect(a, ev_list, Q0, draw_tokens, predict_fn, ct, prod)
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
