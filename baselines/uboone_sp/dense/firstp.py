#!/usr/bin/env python3
"""First-principles diagnostics of the denoiser outputs (post hoc; no training).

    scripts/helix_run.sh python baselines/uboone_sp/dense/firstp.py --stage A --set cal  --out <dir>
    scripts/helix_run.sh python baselines/uboone_sp/dense/firstp.py --stage A --set test --out <dir>
    scripts/helix_run.sh python baselines/uboone_sp/dense/firstp.py --stage B --out <dir>      # needs A(cal), A(test)
    scripts/helix_run.sh python baselines/uboone_sp/dense/firstp.py --stage F --out <dir> --json <results.json>

The models are trained on CASE-CONTROL rows (helix.data.denoise.DenoiseEvents.rows: every
charged cell up to sig_cap, plus empty cells drawn by label-dependent rules), restricted to
token-covered cells, with MSE on y = log1p(q/Q0) and BCE on presence = (training truth q > 0).
Deployment = every token-covered cell of an event.

Stage A (per event): the training-row sample of the SAME recipe (the checkpoints' args:
sig_cap 20000, neg 1, near 1, cov 2, win 48, near_any 1, win_near 24) drawn on this event's
truth (hits truth or complete truth, per model), the deployment cells, and counts of
'charged' under each model's training label per (plane, truth-density stratum,
prediction-density stratum); on calibration events also the log-space residual tables
(y_true - y_hat) per (y_hat bin, prediction-density bin) for the Jensen correction.
Stage B (per test event): presence reliability / Brier / log-loss raw and prior-corrected
(logit' = logit - log(r_train / r_deploy), r = odds of charged; global per plane, per
truth-density stratum [diagnostic: uses the truth], per prediction-density stratum
[deployable]); linear-space charge sums for the estimators raw, gated, hurdle
(p_corrected x conditional charge), each with and without the log-normal Jensen factor;
CRPS of the charge (Gaussian in y with the calibration residual sd; point-mass + Gaussian
mixture for the hurdle models).
Stage F: aggregation with event bootstrap.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, BASE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(BASE, "v2"))
CFS = "/global/cfs/cdirs/m5238/users/oalter"
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
PS_CORPUS, CFS_CORPUS = "/pscratch/sd/o/oalter/coeff_tpc_r1", f"{CFS}/coeff_tpc_r1"
CT_ROOT = f"{CFS}/denoise_truth_complete"
Q0 = 2043.2578108909702
# (name, label truth, hurdle)
MODELS = [("M4n", "h", False), ("M6d", "h", False), ("M6dc", "h", True), ("lb_dec_hits", "h", False),
          ("lb_dec_cond_hits", "h", True), ("lb_dec_ct", "c", False), ("lb_dec_cond_ct", "c", True)]
TEST = list(range(260, 388))
CAL = [e for e in range(0, 64) if e != 41]
DEDGE = np.array([-1, 0.5, 1e4, 1e5, 1e6, 4e6, np.inf]); ND = 6
DLAB = ["iso", "<=10k", "10k-100k", "100k-1M", "1M-4M", ">4M"]
YEDGE = np.array([-np.inf, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, np.inf]); NY = len(YEDGE) - 1
QEDGE = np.array([-1, 0, 25, 102, 500, 1e3, 2e3, 4e3, 8e3, 16e3, 64e3, np.inf]); NQ = len(QEDGE) - 1
QLAB = ["0", "0-25", "25-102", "102-500", "0.5-1k", "1-2k", "2-4k", "4-8k", "8-16k", "16-64k", ">64k"]
TEDGE = np.array([-1, 0, 1e3, 4e3, 16e3, 64e3, 256e3, 1e6, 4e6, np.inf]); NT = len(TEDGE) - 1
TLAB = ["0", "<1k", "1-4k", "4-16k", "16-64k", "64-256k", "256k-1M", "1-4M", ">4M"]
PEDGE = np.array([0, 1e-4, 1e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0 + 1e-9])
NP = len(PEDGE) - 1
LABELS = ["own", "c25", "c102"]
PVAR = ["raw", "glob", "tdens", "pdens"]                 # presence variants
EST = ["raw", "gated", "hurdle", "rawJ", "gatedJ", "hurdleJ", "rawP", "rawPJ"]   # rawP: raw x p_corr / p (prior shift on the charge map)


def src_dir(model, ev):
    if ev >= 260:
        if model == "M4n":
            return f"{CFS}/helix_runs/dense2/model"
        if model in ("M6d", "M6dc"):
            return f"{CFS}/helix_runs/dense3/model"
        return f"{CFS}/helix_runs/firstp/model"
    if model in ("M4n", "M6d", "M6dc"):
        return f"{CFS}/helix_runs/draws/pt_stored"
    return f"{CFS}/helix_runs/firstp/model_cal"


def box(X, r=8):
    P = np.pad(np.asarray(X, np.float64), ((r + 1, r), (r + 1, r)))
    S = P.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]


def dbin(x):
    return np.clip(np.searchsorted(DEDGE, x, side="left") - 1, 0, ND - 1)


def load_event(ev):
    """Truth maps (hits fine cells, complete fine cells), coverage, model maps."""
    import h5py
    from helix.probe.resolution import unkey
    from sp.data import Event, registry
    from sp.pipeline import fine_cells
    reg = registry()
    E = Event(ev)
    shard = E.shard.replace(PS_CORPUS, CFS_CORPUS)
    with h5py.File(shard, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], E.event))
    Th, Tc = {}, {}
    run = os.path.basename(os.path.dirname(shard)); tag = os.path.basename(shard).replace("sim_wire_coeff_", "").replace(".h5", "")
    zc = np.load(os.path.join(CT_ROOT, run, f"truth_{tag}.npz"))
    i = int(np.nonzero(zc["event"] == E.event)[0][0])
    ck, cq = zc["keys"][zc["offsets"][i]:zc["offsets"][i + 1]], zc["q"][zc["offsets"][i]:zc["offsets"][i + 1]].astype(np.float64)
    cg, cw, ct = unkey(ck)
    for g in range(6):
        shape = (reg[g]["n_wires"], reg[g]["n_ticks"])
        Th[g] = fine_cells(E.truth(g, shape)).astype(np.float64)
        M = np.zeros(Th[g].shape)
        m = (cg == g) & (cw < M.shape[0]) & (ct >= 0) & (ct < M.shape[1])
        np.add.at(M, (cw[m], ct[m]), cq[m])
        Tc[g] = M
    maps = {}
    for name, _, _ in MODELS:
        f = os.path.join(src_dir(name, ev), f"ev{ev:03d}.npz")
        maps[name] = np.load(f)
    return E, shard, pos, Th, Tc, maps


def model_planes(Mz, name, g, shape):
    to = int(Mz["toff"])
    crop = lambda X: X[:shape[0], to:to + shape[1]].astype(np.float64)
    return crop(Mz[f"q_{name}_g{g}"]), crop(Mz[f"p_{name}_g{g}"]), crop(Mz[f"cov_g{g}"]) > 0


def training_rows(shard, pos, T, cfg, seed):
    """DenoiseEvents.rows on this event with the checkpoints' recipe (truth T: {gid: fine map})."""
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.probe.resolution import fkey
    ds = DenoiseEvents("", [], "", items=[], cov_per_sig=2.0, win_per_event=48, near_any_per_sig=1.0, win_near_per_event=24)
    ce = read_coeff_event(shard, pos)
    K, Q = [], []
    for g in range(6):
        fw, ft = np.nonzero(T[g] > 0)
        K.append(fkey(np.full(len(fw), g), fw, ft)); Q.append(T[g][fw, ft])
    k = np.concatenate(K); o = np.argsort(k)
    rows, _ = ds.rows(ce, k[o], np.concatenate(Q)[o].astype(np.float32), np.random.default_rng(seed))
    return rows


def stage_a(job):
    ev, out_dir = job
    out = os.path.join(out_dir, f"A_ev{ev:03d}.npz")
    if os.path.exists(out):
        return ev, "exists"
    from helix.probe.resolution import unkey
    E, shard, pos, Th, Tc, maps = load_event(ev)
    rows = {"h": training_rows(shard, pos, Th, None, 11 + ev), "c": training_rows(shard, pos, Tc, None, 23 + ev)}
    res = {}
    for lab in ("h", "c"):
        g_, fw, ft = unkey(rows[lab])
        res[f"rows_{lab}"] = (g_, fw, ft)
    # counts: [model][plane, tdens, pdens, (n_dep, pos_dep, n_rows, pos_rows)]
    C = np.zeros((len(MODELS), 3, ND, ND, 4))
    # residual tables (all covered cells / charged only): [model, ybin, pdens, (n, s, s2, lin_true, lin_pred)]
    RT = np.zeros((len(MODELS), 2, NY, ND, 5))
    for g in range(6):
        p_ = g % 3
        shape = Th[g].shape
        td = dbin(np.where(Th[g] > 0, box(Th[g]) - Th[g], box(Th[g])))
        for mi, (name, lab, hurdle) in enumerate(MODELS):
            q, p, cov = model_planes(maps[name], name, g, shape)
            T = Th[g] if lab == "h" else Tc[g]
            y = T > 0
            pdn = dbin(box(q) - q)
            gr, wr, tr = res[f"rows_{lab}"]
            m = (gr == g) & (wr < shape[0]) & (tr >= 0) & (tr < shape[1])
            R = np.zeros(shape, bool); R[wr[m], tr[m]] = True
            R &= cov
            for sel, j in ((cov, 0), (R, 2)):
                idx = (td[sel] * ND + pdn[sel])
                C[mi, p_, :, :, j] += np.bincount(idx, minlength=ND * ND).reshape(ND, ND)
                C[mi, p_, :, :, j + 1] += np.bincount(idx, weights=y[sel].astype(float), minlength=ND * ND).reshape(ND, ND)
            yh = np.log1p(q / Q0); yt = np.log1p(T / Q0)
            for k_, sel in ((0, cov), (1, cov & y)):
                yb = np.clip(np.searchsorted(YEDGE, yh[sel], side="right") - 1, 0, NY - 1)
                idx = yb * ND + pdn[sel]
                r = yt[sel] - yh[sel]
                for j, w in enumerate((None, r, r * r, T[sel], q[sel])):
                    RT[mi, k_, :, :, j] += np.bincount(idx, weights=w, minlength=NY * ND).reshape(NY, ND)
    np.savez(out + ".tmp.npz", C=C, RT=RT)
    os.replace(out + ".tmp.npz", out)
    return ev, "ok"


def tables(out_dir, evs):
    A = [np.load(os.path.join(out_dir, f"A_ev{e:03d}.npz")) for e in evs if os.path.exists(os.path.join(out_dir, f"A_ev{e:03d}.npz"))]
    C = sum(a["C"] for a in A); RT = sum(a["RT"] for a in A)
    return C, RT, len(A)


def logodds(pos, n):
    return np.log((pos + 0.5) / (n - pos + 0.5))


def corrections(C):
    """log(r_train / r_deploy) per model: glob [model, plane], tdens [model, plane, td], pdens [model, plane, pd]."""
    dep_n, dep_p, row_n, row_p = (C[..., j] for j in range(4))
    glob_ = logodds(row_p.sum((2, 3)), row_n.sum((2, 3))) - logodds(dep_p.sum((2, 3)), dep_n.sum((2, 3)))
    td = logodds(row_p.sum(3), row_n.sum(3)) - logodds(dep_p.sum(3), dep_n.sum(3))
    pd = logodds(row_p.sum(2), row_n.sum(2)) - logodds(dep_p.sum(2), dep_n.sum(2))
    return glob_, td, pd


def sigma_tables(RT):
    """log-space residual mean / variance per (model, all|charged, ybin, pdens), with fallbacks to the y-bin marginal."""
    n, s, s2 = RT[..., 0], RT[..., 1], RT[..., 2]
    nm, sm, s2m = n.sum(3, keepdims=True), s.sum(3, keepdims=True), s2.sum(3, keepdims=True)
    use = n >= 200
    n_, s_, s2_ = np.where(use, n, nm), np.where(use, s, sm), np.where(use, s2, s2m)
    mu = s_ / np.maximum(n_, 1)
    var = np.maximum(s2_ / np.maximum(n_, 1) - mu ** 2, 0)
    return mu, var


def crps_gauss(mu, sd, y):
    from scipy.special import erf
    z = (y - mu) / np.maximum(sd, 1e-6)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi); Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
    return sd * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def crps_mixture(p, mu, sd, y):
    """CRPS of (1-p) delta_0 + p N(mu, sd) at y (y >= 0)."""
    from scipy.special import erf
    sd = np.maximum(sd, 1e-6)

    def e_abs(a):                                       # E|N - a|
        d = mu - a; z = d / sd
        return d * erf(z / np.sqrt(2)) + 2 * sd * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    e_xy = (1 - p) * np.abs(y) + p * e_abs(y)
    e_xx = 2 * p * (1 - p) * e_abs(0.0) + p * p * 2 * sd / np.sqrt(np.pi)
    return e_xy - 0.5 * e_xx


def stage_b(job):
    ev, out_dir, corr, sig = job
    out = os.path.join(out_dir, f"B_ev{ev:03d}.npz")
    if os.path.exists(out):
        return ev, "exists"
    glob_, tdc, pdc = corr
    mu_t, var_t = sig
    E, shard, pos, Th, Tc, maps = load_event(ev)
    M = len(MODELS)
    # reliability: [model, pvar, label, plane, td, pbin, (n, sum p, sum y)]
    REL = np.zeros((M, len(PVAR), len(LABELS), 3, ND, NP, 3))
    # proper scores (own label): [model, pvar, plane, td, (n, brier, logloss)]
    SC = np.zeros((M, len(PVAR), 3, ND, 3))
    CR = np.zeros((M, 2, 3, ND, 2))                       # CRPS: [model, (gauss, mixture), plane, td, (n, sum)]
    # linear sums vs truth t in {hits, complete}: cells [model, t, plane, qbin, est+truth]
    LC = np.zeros((M, 2, 3, NQ, len(EST) + 1))
    LT4 = np.zeros((M, 2, 3, NT, len(EST) + 1)); LT16 = np.zeros((M, 2, 3, NT, len(EST) + 1))
    LP = np.zeros((M, 2, 3, len(EST) + 1))                 # plane totals
    # binned by the ESTIMATOR's own value (E[truth | prediction]): [model, t, plane, est, bin, (n, sum pred, sum truth)]
    LQ = np.zeros((M, 2, 3, len(EST), NQ, 3)); LTP = np.zeros((M, 2, 3, len(EST), NT, 3))
    for g in range(6):
        p_ = g % 3
        shape = Th[g].shape
        td = dbin(np.where(Th[g] > 0, box(Th[g]) - Th[g], box(Th[g])))
        for mi, (name, lab, hurdle) in enumerate(MODELS):
            q, p, cov = model_planes(maps[name], name, g, shape)
            T = Th[g] if lab == "h" else Tc[g]
            pdn = dbin(box(q) - q)
            lg = np.log(np.clip(p, 1e-7, 1 - 1e-7)) - np.log1p(-np.clip(p, 1e-7, 1 - 1e-7))
            pv = {"raw": p, "glob": 1 / (1 + np.exp(-(lg - glob_[mi, p_]))),
                  "tdens": 1 / (1 + np.exp(-(lg - tdc[mi, p_][td]))), "pdens": 1 / (1 + np.exp(-(lg - pdc[mi, p_][pdn])))}
            labs = {"own": T > 0, "c25": Tc[g] > 25, "c102": Tc[g] > 102}
            tdv = td[cov]
            for vi, v in enumerate(PVAR):
                pc = pv[v][cov]
                pb = np.clip(np.searchsorted(PEDGE, pc, side="right") - 1, 0, NP - 1)
                for li, l in enumerate(LABELS):
                    yl = labs[l][cov].astype(float)
                    idx = tdv * NP + pb
                    for j, w in enumerate((None, pc, yl)):
                        REL[mi, vi, li, p_, :, :, j] += np.bincount(idx, weights=w, minlength=ND * NP).reshape(ND, NP)
                yo = labs["own"][cov].astype(float)
                pcc = np.clip(pc, 1e-7, 1 - 1e-7)
                ll = -(yo * np.log(pcc) + (1 - yo) * np.log(1 - pcc))
                for j, w in enumerate((None, (pc - yo) ** 2, ll)):
                    SC[mi, vi, p_, :, j] += np.bincount(tdv, weights=w, minlength=ND)
            # charge estimators (covered cells; uncovered cells predict 0 and still count in the truth sums)
            yh = np.log1p(q / Q0)
            yb = np.clip(np.searchsorted(YEDGE, yh, side="right") - 1, 0, NY - 1)
            k_ = 1 if hurdle else 0
            var = var_t[mi, k_][yb, pdn]
            qJ = Q0 * np.expm1(yh + var / 2) * (q > 0)
            pc = pv["pdens"]
            est = {"raw": q, "gated": q * (p > 0.5), "hurdle": pc * q if hurdle else np.full(shape, np.nan),
                   "rawJ": qJ, "gatedJ": qJ * (p > 0.5), "hurdleJ": pc * qJ if hurdle else np.full(shape, np.nan),
                   "rawP": q * pc / np.maximum(p, 1e-7) if not hurdle else np.full(shape, np.nan),
                   "rawPJ": qJ * pc / np.maximum(p, 1e-7) if not hurdle else np.full(shape, np.nan)}
            for e_ in est.values():
                e_[~cov] = 0.0
            # CRPS in y (own label truth): Gaussian with the all-cell residual sd; mixture for the hurdle
            yt = np.log1p(T / Q0)
            sd_all = np.sqrt(var_t[mi, 0][yb, pdn] + 1e-6)
            CR[mi, 0, p_, :, 0] += np.bincount(tdv, minlength=ND)
            CR[mi, 0, p_, :, 1] += np.bincount(tdv, weights=crps_gauss(yh + mu_t[mi, 0][yb, pdn], sd_all, yt)[cov], minlength=ND)
            if hurdle:
                sd_c = np.sqrt(var_t[mi, 1][yb, pdn] + 1e-6)
                CR[mi, 1, p_, :, 0] += np.bincount(tdv, minlength=ND)
                CR[mi, 1, p_, :, 1] += np.bincount(tdv, weights=crps_mixture(pc, yh + mu_t[mi, 1][yb, pdn], sd_c, yt)[cov], minlength=ND)
            for ti, Tt in enumerate((Th[g], Tc[g])):
                qb = np.clip(np.searchsorted(QEDGE, Tt, side="left") - 1, 0, NQ - 1)
                for j, nm in enumerate(EST):
                    LC[mi, ti, p_, :, j] += np.bincount(qb.ravel(), weights=np.nan_to_num(est[nm]).ravel(), minlength=NQ)
                    LP[mi, ti, p_, j] += np.nan_to_num(est[nm]).sum()
                LC[mi, ti, p_, :, -1] += np.bincount(qb.ravel(), weights=Tt.ravel(), minlength=NQ)
                LP[mi, ti, p_, -1] += Tt.sum()
                for j, nm in enumerate(EST):
                    ev_ = np.nan_to_num(est[nm])
                    if np.isnan(est[nm]).all():
                        continue
                    pb_ = np.clip(np.searchsorted(QEDGE, ev_, side="left") - 1, 0, NQ - 1).ravel()
                    for k2, w in enumerate((None, ev_.ravel(), Tt.ravel())):
                        LQ[mi, ti, p_, j, :, k2] += np.bincount(pb_, weights=w, minlength=NQ)
                    nw, nt = (shape[0] // 16) * 16, (shape[1] // 8) * 8
                    t16 = lambda X: X[:nw, :nt].reshape(nw // 16, 16, nt // 8, 8).sum((1, 3)).ravel()
                    pt_, tt_ = t16(ev_), t16(Tt)
                    tb_ = np.clip(np.searchsorted(TEDGE, pt_, side="left") - 1, 0, NT - 1)
                    for k2, w in enumerate((None, pt_, tt_)):
                        LTP[mi, ti, p_, j, :, k2] += np.bincount(tb_, weights=w, minlength=NT)
                for a_, b_, L in ((4, 4, LT4), (16, 8, LT16)):
                    nw, nt = (shape[0] // a_) * a_, (shape[1] // b_) * b_
                    til = lambda X: X[:nw, :nt].reshape(nw // a_, a_, nt // b_, b_).sum((1, 3)).ravel()
                    tt = til(Tt)
                    tb = np.clip(np.searchsorted(TEDGE, tt, side="left") - 1, 0, NT - 1)
                    for j, nm in enumerate(EST):
                        L[mi, ti, p_, :, j] += np.bincount(tb, weights=til(np.nan_to_num(est[nm])), minlength=NT)
                    L[mi, ti, p_, :, -1] += np.bincount(tb, weights=tt, minlength=NT)
    np.savez(out + ".tmp.npz", REL=REL, SC=SC, CR=CR, LC=LC, LT4=LT4, LT16=LT16, LP=LP, LQ=LQ, LTP=LTP)
    os.replace(out + ".tmp.npz", out)
    return ev, "ok"


def ece(rel):
    """rel [..., NP, (n, sum p, sum y)] -> ECE."""
    n, sp, sy = rel[..., 0], rel[..., 1], rel[..., 2]
    return (np.abs(sp - sy).sum(-1)) / np.maximum(n.sum(-1), 1)


def stage_f(out_dir, out_json):
    Cc, RTc, nc = tables(out_dir, CAL)
    Ct, RTt, nt = tables(out_dir, TEST)
    gc, tdc, pdc = corrections(Cc); gt, tdt, pdt = corrections(Ct)
    B = [np.load(os.path.join(out_dir, f"B_ev{e:03d}.npz")) for e in TEST]
    S = {k: np.stack([b[k] for b in B]) for k in B[0].files}
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(B), len(B)), minlength=len(B)) for _ in range(200)]).astype(float)
    res = {"n_cal": nc, "n_test": nt, "models": [m[0] for m in MODELS]}

    def bt(f, arr):
        c = f(arr.sum(0)); b = np.array([f(np.tensordot(w, arr, 1)) for w in W])
        return [float(c), float(np.percentile(b, 16)), float(np.percentile(b, 84))]
    # ratios
    dep_n, dep_p, row_n, row_p = (Cc[..., j] for j in range(4))
    for mi, (m, lab, hur) in enumerate(MODELS):
        R = res.setdefault(m, {"label": lab, "hurdle": hur})
        R["log_ratio_glob_cal"] = gc[mi].tolist(); R["log_ratio_glob_test"] = gt[mi].tolist()
        R["log_ratio_tdens_cal"] = tdc[mi].tolist(); R["log_ratio_pdens_cal"] = pdc[mi].tolist()
        R["base_rate_dep"] = (dep_p[mi].sum((1, 2)) / dep_n[mi].sum((1, 2))).tolist()
        R["base_rate_rows"] = (row_p[mi].sum((1, 2)) / row_n[mi].sum((1, 2))).tolist()
        R["base_rate_dep_tdens"] = (dep_p[mi].sum((0, 2)) / np.maximum(dep_n[mi].sum((0, 2)), 1)).tolist()
        R["base_rate_rows_tdens"] = (row_p[mi].sum((0, 2)) / np.maximum(row_n[mi].sum((0, 2)), 1)).tolist()
        # proper scores per presence variant (all planes; per plane)
        for vi, v in enumerate(PVAR):
            sc = S["SC"][:, mi, vi]                       # (ev, plane, td, 3)
            R[f"brier_{v}"] = bt(lambda a: a[..., 1].sum() / a[..., 0].sum(), sc)
            R[f"logloss_{v}"] = bt(lambda a: a[..., 2].sum() / a[..., 0].sum(), sc)
            for p_ in range(3):
                R[f"logloss_{v}_{'UVY'[p_]}"] = float(sc[:, p_, :, 2].sum() / sc[:, p_, :, 0].sum())
            for li, l in enumerate(LABELS):
                rel = S["REL"][:, mi, vi, li]               # (ev, plane, td, NP, 3)
                R[f"ece_{v}_{l}"] = bt(lambda a: ece(a.sum((0, 1))), rel)
                for d in range(ND):
                    R[f"ece_{v}_{l}_{DLAB[d]}"] = float(ece(rel[:, :, d].sum((0, 1))))
                for p_ in range(3):
                    R[f"ece_{v}_{l}_{'UVY'[p_]}"] = float(ece(rel[:, p_].sum((0, 1))))
                if v in ("raw", "pdens", "glob") and l == "own":
                    a = rel.sum((0, 1, 2))
                    R[f"reliab_{v}"] = dict(p_mean=(a[:, 1] / np.maximum(a[:, 0], 1)).tolist(), freq=(a[:, 2] / np.maximum(a[:, 0], 1)).tolist(), n=a[:, 0].tolist())
        cr = S["CR"][:, mi]
        R["crps_gauss"] = bt(lambda a: a[0, ..., 1].sum() / a[0, ..., 0].sum(), cr)
        if hur:
            R["crps_mixture"] = bt(lambda a: a[1, ..., 1].sum() / a[1, ..., 0].sum(), cr)
        for d in range(ND):
            R[f"crps_gauss_{DLAB[d]}"] = float(cr[:, 0, :, d, 1].sum() / max(cr[:, 0, :, d, 0].sum(), 1))
        # linear sums: ratio sum(pred)/sum(truth) per truth bin; for the empty bin pred per (hits) truth total
        for ti, tn in enumerate(("hits", "ct")):
            for key, arr, labs in (("cells", S["LC"], QLAB), ("t4", S["LT4"], TLAB), ("t16", S["LT16"], TLAB)):
                a = arr[:, mi, ti]                          # (ev, plane, bin, est+1)
                for j, est in enumerate(EST):
                    if (est.startswith("hurdle") and not hur) or (est.startswith("rawP") and hur):
                        continue
                    for b in range(a.shape[2]):
                        if b == 0:                          # empty: predicted charge in empty cells/tiles per unit total truth
                            R[f"lin_{tn}_{key}_{est}_{labs[b]}"] = bt(lambda x, b=b, j=j: x[:, b, j].sum() / x[:, :, -1].sum(), a)
                        else:
                            R[f"lin_{tn}_{key}_{est}_{labs[b]}"] = bt(lambda x, b=b, j=j: x[:, b, j].sum() / x[:, b, -1].sum(), a)
                            for p_ in range(3):
                                R[f"lin_{tn}_{key}_{est}_{labs[b]}_{'UVY'[p_]}"] = float(a[:, p_, b, j].sum() / max(a[:, p_, b, -1].sum(), 1e-9))
            lp = S["LP"][:, mi, ti]
            for j, est in enumerate(EST):
                if (est.startswith("hurdle") and not hur) or (est.startswith("rawP") and hur):
                    continue
                R[f"plane_{tn}_{est}"] = bt(lambda x, j=j: x[:, j].sum() / x[:, -1].sum(), lp)
                for p_ in range(3):
                    R[f"plane_{tn}_{est}_{'UVY'[p_]}"] = float(lp[:, p_, j].sum() / lp[:, p_, -1].sum())
        for ti, tn in enumerate(("hits", "ct")):
            for key, arr, labs in (("bypred_cells", S["LQ"], QLAB), ("bypred_t16", S["LTP"], TLAB)):
                a = arr[:, mi, ti]                          # (ev, plane, est, bin, 3)
                for j, est in enumerate(EST):
                    if (est.startswith("hurdle") and not hur) or (est.startswith("rawP") and hur):
                        continue
                    for b in range(a.shape[3]):
                        if a[:, :, j, b, 0].sum() < 30:
                            continue
                        # truth / prediction in the prediction bin (1 = unbiased); bin 0: truth carried by cells predicted 0, per total truth
                        if b == 0:
                            R[f"{key}_{tn}_{est}_{labs[b]}"] = bt(lambda x, b=b, j=j: x[:, j, b, 2].sum() / x[:, j, :, 2].sum(), a)
                        else:
                            R[f"{key}_{tn}_{est}_{labs[b]}"] = bt(lambda x, b=b, j=j: x[:, j, b, 2].sum() / x[:, j, b, 1].sum(), a)
        mu, var = sigma_tables(RTc)
        R["sigma_log_all_by_ybin"] = np.sqrt((RTc[mi, 0, :, :, 2].sum(1) / np.maximum(RTc[mi, 0, :, :, 0].sum(1), 1)) -
                                             (RTc[mi, 0, :, :, 1].sum(1) / np.maximum(RTc[mi, 0, :, :, 0].sum(1), 1)) ** 2).tolist()
        R["n_by_ybin"] = RTc[mi, 0, :, :, 0].sum(1).tolist()
    json.dump(res, open(out_json, "w"), indent=0)
    print("wrote", out_json)


def main():
    sys.path.insert(0, os.path.join(BASE, "v2"))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("A", "B", "F"))
    ap.add_argument("--set", default="test", choices=("cal", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from multiprocessing import get_context
    if a.stage == "A":
        evs = CAL if a.set == "cal" else TEST
        with get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
            for ev, msg in pool.imap_unordered(stage_a, [(e, a.out) for e in evs]):
                print(f"A ev{ev:03d}: {msg}", flush=True)
    elif a.stage == "B":
        Cc, RTc, nc = tables(a.out, CAL)
        corr = corrections(Cc)
        sig = sigma_tables(RTc)
        print(f"calibration events {nc}; glob log-ratio per model/plane:\n{np.round(corr[0], 2)}", flush=True)
        with get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:
            for ev, msg in pool.imap_unordered(stage_b, [(e, a.out, corr, sig) for e in TEST]):
                print(f"B ev{ev:03d}: {msg}", flush=True)
    else:
        stage_f(a.out, a.json)


if __name__ == "__main__":
    main()
