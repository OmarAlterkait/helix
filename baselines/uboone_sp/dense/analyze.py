#!/usr/bin/env python3
"""Stage 2 of the dense-activity comparison: metrics with event-bootstrap intervals.

    scripts/helix_run.sh python baselines/uboone_sp/dense/analyze.py \
        --rec /pscratch/sd/o/oalter/helix_work/uboone_sp/dense/rec_fit \
        --out baselines/uboone_sp/dense/results/dense_fit.json

Every interval is the 16-84% range over B event-bootstrap resamples (events drawn
with replacement, each resample a multinomial weight per event). Central values
are computed on the full sample; quantiles / thresholds use fine histograms
(bootstrap) whose bin width is below the quoted precision.

Strata (from collect.py):
  density  true charge within +-8 cells (17 x 17 cells = 34 wires x 272 ticks), the
           cell excluded. Bins (e-): isolated (0), <=1e4, 1e4-1e5, 1e5-1e6, 1e6-4e6,
           >4e6. One MIP crossing the box is ~0.4-1e6 e-.
  charge   cell truth: (0), <500, 0.5-1k, 1-2k, 2-4k, 4-8k, 8-16k, 16-64k, >64k e-.
"""
import argparse
import glob
import json
import os

import numpy as np

Q0 = 2043.258
DB = [(-1, 0.5), (0.5, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 4e6), (4e6, 1e12)]
DLAB = ["iso", "<=10k", "10k-100k", "100k-1M", "1M-4M", ">4M"]
QB = [(0, 500), (500, 1e3), (1e3, 2e3), (2e3, 4e3), (4e3, 8e3), (8e3, 16e3), (16e3, 64e3), (64e3, 1e12)]
QLAB = ["<500", "0.5-1k", "1-2k", "2-4k", "4-8k", "8-16k", "16-64k", ">64k"]
PL = ["U", "V", "Y", "all"]
B = 200
REDGE = np.concatenate([[-1e-9, 1e-9], np.logspace(-3, 2, 1001)])         # pred/true: exact 0, then log 1e-3..1e2
AEDGE = np.concatenate([[-1e-9, 1e-9], np.logspace(0, 7, 701)])           # |pred - true|, e-


def load(rec):
    C = json.load(open(os.path.join(rec, "columns.json")))
    files = sorted(glob.glob(os.path.join(rec, "ev*.npz")))
    D = {k: [] for k in ("sig", "emp", "reg16", "reg4", "par", "sep", "evt")}
    fc, rf16, rf4 = [], [], []
    for i, f in enumerate(files):
        z = np.load(f)
        for k in D:
            x = z[k]
            D[k].append(np.concatenate([x, np.full((len(x), 1), i, x.dtype)], 1))
        fc.append(z["fc"]); rf16.append(z["regfc16"]); rf4.append(z["regfc4"])
    D = {k: np.concatenate(v) for k, v in D.items()}
    return C, D, np.stack(fc), np.stack(rf16), np.stack(rf4), len(files)


def boot_weights(nev, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, nev, nev), minlength=nev) for _ in range(B)]).astype(np.float64)


def col(C, D, key, name):
    return D[key][:, C[f"{key}_cols"].index(name)].astype(np.float64)


def dbin(d):
    out = np.full(len(d), -1)
    for i, (lo, hi) in enumerate(DB):
        out[(d > lo) & (d <= hi)] = i
    return out


def qbin(t):
    out = np.full(len(t), -1)
    for i, (lo, hi) in enumerate(QB):
        out[(t > lo) & (t <= hi)] = i
    return out


def hist_q(H, edges, qs):
    """Quantiles from histograms H (..., nb) on bin edges (linear interpolation inside a bin)."""
    c = np.cumsum(H, -1)
    tot = c[..., -1:]
    out = []
    for q in qs:
        k = (c < q * tot).sum(-1)
        k = np.minimum(k, H.shape[-1] - 1)
        lo, hi = edges[k], edges[k + 1]
        cb = np.take_along_axis(c, k[..., None], -1)[..., 0]
        hb = np.take_along_axis(H, k[..., None], -1)[..., 0]
        f = np.clip((q * tot[..., 0] - (cb - hb)) / np.maximum(hb, 1e-12), 0, 1)
        out.append(lo + f * (hi - lo))
    return out


def grouped_hist(vals, g, ev, ng, nev, edges, w=None):
    k = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    ok = g >= 0
    idx = (g[ok] * nev + ev[ok]) * nb + k[ok]
    return np.bincount(idx, weights=None if w is None else w[ok], minlength=ng * nev * nb).reshape(ng, nev, nb)


def summarise(Hge, W, edges, fn):
    """Hge: (ng, nev, nb). fn(H (..., nb)) -> array (...). Returns central (ng,), lo, hi (ng,)."""
    cen = fn(Hge.sum(1))
    bs = fn(np.einsum("be,gek->gbk", W, Hge))
    return cen, np.percentile(bs, 16, 1), np.percentile(bs, 84, 1)


def iv(c, lo, hi, i, nd=3):
    return [round(float(c[i]), nd), round(float(lo[i]), nd), round(float(hi[i]), nd)]


# ---------------------------------------------------------------- 1. fidelity
def fidelity(C, D, W, nev):
    sig = D["sig"]
    ev = sig[:, -1].astype(int)
    p, T = col(C, D, "sig", "plane").astype(int), col(C, D, "sig", "T")
    di, qi = dbin(col(C, D, "sig", "dens_q")), qbin(T)
    nd, nq = len(DB), len(QB)
    res = {}
    counts = {}
    for pp in range(4):
        mp = (p == pp) if pp < 3 else np.ones(len(p), bool)
        for a in range(nd):
            for b in range(nq):
                counts[f"{PL[pp]}|{DLAB[a]}|{QLAB[b]}"] = int((mp & (di == a) & (qi == b)).sum())
    res["counts"] = counts
    for m in C["charge"]:
        X = col(C, D, "sig", m)
        r = X / T
        ae = np.abs(X - T)
        out = {}
        for pp in range(4):
            mp = (p == pp) if pp < 3 else np.ones(len(p), bool)
            g = np.where(mp & (di >= 0) & (qi >= 0), di * nq + qi, -1)
            # also density-marginal (all charges) rows: groups nd*nq + a
            g2 = np.where(mp & (di >= 0), nd * nq + di, -1)
            g3 = np.where(mp & (qi >= 0), nd * nq + nd + qi, -1)
            ng = nd * nq + nd + nq
            Hr = sum(grouped_hist(r, gg, ev, ng, nev, REDGE) for gg in (g, g2, g3))
            Ha = sum(grouped_hist(ae / T, gg, ev, ng, nev, REDGE) for gg in (g, g2, g3))
            med = summarise(Hr, W, REDGE, lambda H: hist_q(H, REDGE, [0.5])[0])
            hw = summarise(Hr, W, REDGE, lambda H: (lambda a: (a[1] - a[0]) / 2)(hist_q(H, REDGE, [0.16, 0.84])))
            fz = summarise(Hr, W, REDGE, lambda H: H[..., 0] / np.maximum(H.sum(-1), 1))
            rae = summarise(Ha, W, REDGE, lambda H: hist_q(H, REDGE, [0.5])[0])
            n = Hr.sum((1, 2))
            for gi in range(ng):
                if n[gi] < 30:
                    continue
                if gi < nd * nq:
                    key = f"{DLAB[gi // nq]}|{QLAB[gi % nq]}"
                elif gi < nd * nq + nd:
                    key = f"{DLAB[gi - nd * nq]}|all"
                else:
                    key = f"all|{QLAB[gi - nd * nq - nd]}"
                out[f"{PL[pp]}|{key}"] = dict(n=int(n[gi]), median_ratio=iv(*med, gi), hw68=iv(*hw, gi),
                                              frac_zero=iv(*fz, gi), median_rel_abs_err=iv(*rae, gi))
        res[m] = out
    return res


# ---------------------------------------------------------------- 2. false charge on empty cells
def false_charge(C, fc, W):
    edges = np.array(C["fc_edges"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    centers[0] = 0
    di = dbin(centers)
    res = {}
    for i, m in enumerate(C["charge"]):
        for pp in range(4):
            F = fc[:, i, pp] if pp < 3 else fc[:, i].sum(1)                      # (nev, nfine, 4)
            for a in range(len(DB)):
                Fa = F[:, di == a].sum(1)                                          # (nev, 4)
                if Fa[:, 0].sum() < 100:
                    continue
                cen = Fa.sum(0)
                bs = W @ Fa
                d = {"n_empty": int(cen[0])}
                for k, lab in enumerate(("gt005Q0", "gt025Q0", "gt1Q0")):
                    fr = bs[:, k + 1] / bs[:, 0]
                    d[lab] = [float(cen[k + 1] / cen[0]), float(np.percentile(fr, 16)), float(np.percentile(fr, 84))]
                    d[f"count_{lab}"] = int(cen[k + 1])
                res[f"{m}|{PL[pp]}|{DLAB[a]}"] = d
    return res


def false_charge_quantiles(C, D):
    """Weighted quantiles of predicted charge on sampled empty cells, by density (no bootstrap)."""
    emp = D["emp"]
    p, w = col(C, D, "emp", "plane").astype(int), col(C, D, "emp", "w")
    di = dbin(col(C, D, "emp", "dens_q"))
    res = {}
    for m in C["charge"]:
        X = col(C, D, "emp", m)
        for pp in range(4):
            mp = (p == pp) if pp < 3 else np.ones(len(p), bool)
            for a in range(len(DB)):
                s = mp & (di == a)
                if s.sum() < 100:
                    continue
                o = np.argsort(X[s]); cw = np.cumsum(w[s][o]); cw /= cw[-1]
                qv = [float(X[s][o][min(np.searchsorted(cw, q), s.sum() - 1)]) for q in (0.5, 0.9, 0.99, 0.999)]
                res[f"{m}|{PL[pp]}|{DLAB[a]}"] = dict(n_sampled=int(s.sum()), q50_90_99_999=qv,
                                                      mean=float(np.sum(X[s] * w[s]) / np.sum(w[s])))
    return res


# ---------------------------------------------------------------- 3. presence: ROC / eff at context-matched FPR
def presence(C, D, W, nev, fpr=0.01, nbs=1500):
    sig, emp = D["sig"], D["emp"]
    evs, eve = sig[:, -1].astype(int), emp[:, -1].astype(int)
    ps, pe = col(C, D, "sig", "plane").astype(int), col(C, D, "emp", "plane").astype(int)
    ds, de = dbin(col(C, D, "sig", "dens_q")), dbin(col(C, D, "emp", "dens_q"))
    qs = qbin(col(C, D, "sig", "T"))
    we = col(C, D, "emp", "w")
    nd, nq = len(DB), len(QB)
    res = {}
    for m in C["methods"]:
        xs, xe = col(C, D, "sig", m), col(C, D, "emp", m)
        allv = np.concatenate([xs, xe])
        nz = allv[allv != 0]
        e = np.unique(np.concatenate([np.quantile(nz, np.linspace(0, 1, nbs)), [0.0, 1e-12]]))
        e = np.concatenate([[-np.inf], e, [np.inf]])
        # empties per (plane, dbin); charged per (plane, dbin, qbin) and (plane, dbin, all)
        He = grouped_hist(xe, np.where(de >= 0, pe * nd + de, -1), eve, 3 * nd, nev, e, we)
        Hs = grouped_hist(xs, np.where((ds >= 0) & (qs >= 0), (ps * nd + ds) * (nq + 1) + qs, -1), evs,
                          3 * nd * (nq + 1), nev, e)
        Hs = Hs.reshape(3, nd, nq + 1, nev, -1)
        Hs[:, :, nq] = Hs[:, :, :nq].sum(2)
        He = He.reshape(3, nd, nev, -1)

        def metrics(He_, Hs_):
            """He_ (L, 3, nd, nb), Hs_ (L, 3, nd, nq+1, nb) -> eff, auc (L, 4, nd, nq+1), achieved fpr (L, 3, nd)."""
            ce = np.cumsum(He_[..., ::-1], -1)[..., ::-1]                    # empties at or above bin k
            tot_e = He_.sum(-1, keepdims=True)
            fr = ce / np.maximum(tot_e, 1e-12)
            okk = fr <= fpr
            k = np.where(okk.any(-1), np.argmax(okk, -1), He_.shape[-1] - 1)  # first bin whose upper tail <= fpr
            afpr = np.take_along_axis(fr, k[..., None], -1)[..., 0]
            cs = np.cumsum(Hs_[..., ::-1], -1)[..., ::-1]
            tot_s = Hs_.sum(-1)
            kk = np.broadcast_to(k[..., None, None], cs.shape[:-1] + (1,))
            above = np.take_along_axis(cs, kk, -1)[..., 0]
            L = He_.shape[0]
            eff = np.zeros((L, 4) + tot_s.shape[2:]); auc = np.zeros_like(eff)
            eff[:, :3] = above / np.maximum(tot_s, 1e-12)
            eff[:, 3] = above.sum(1) / np.maximum(tot_s.sum(1), 1e-12)        # per-plane thresholds, pooled
            below_e = np.cumsum(He_, -1) - He_                                # empties strictly below bin
            num = (Hs_ * (below_e[:, :, :, None] + 0.5 * He_[:, :, :, None])).sum(-1)
            auc[:, :3] = num / np.maximum(tot_s * tot_e, 1e-12)
            Hep, Hsp = He_.sum(1), Hs_.sum(1)                                 # pooled AUC (single scale)
            be = np.cumsum(Hep, -1) - Hep
            auc[:, 3] = (Hsp * (be[:, :, None] + 0.5 * Hep[:, :, None])).sum(-1) / np.maximum(
                Hsp.sum(-1) * Hep.sum(-1)[:, :, None], 1e-12)
            return eff, auc, afpr

        eff, auc, afpr = (x[0] for x in metrics(He.sum(2)[None], Hs.sum(3)[None]))
        nb_ = He.shape[-1]
        Heb = np.moveaxis(He, 2, 0).reshape(nev, -1)
        Hsb = np.moveaxis(Hs, 3, 0).reshape(nev, -1)
        effb, aucb = [], []
        for b0 in range(0, B, 50):
            Wb = W[b0:b0 + 50]
            e_, a_, _ = metrics((Wb @ Heb).reshape(len(Wb), 3, nd, nb_), (Wb @ Hsb).reshape(len(Wb), 3, nd, nq + 1, nb_))
            effb.append(e_); aucb.append(a_)
        effb, aucb = np.concatenate(effb), np.concatenate(aucb)
        ntot = Hs.sum((3, 4))
        for pp in range(4):
            for a in range(nd):
                for b in range(nq + 1):
                    n = int(ntot[:, a, b].sum() if pp == 3 else ntot[pp, a, b])
                    if n < 30:
                        continue
                    k = f"{m}|{PL[pp]}|{DLAB[a]}|{QLAB[b] if b < nq else 'all'}"
                    res[k] = dict(n=n,
                                  eff=[float(eff[pp, a, b]), float(np.percentile(effb[:, pp, a, b], 16)), float(np.percentile(effb[:, pp, a, b], 84))],
                                  auc=[float(auc[pp, a, b]), float(np.percentile(aucb[:, pp, a, b], 16)), float(np.percentile(aucb[:, pp, a, b], 84))])
                    if pp < 3:
                        res[k]["achieved_fpr"] = float(afpr[pp, a])
    return res


# ---------------------------------------------------------------- 4. regions
RQB = [(-1, 0), (0, 1e3), (1e3, 4e3), (4e3, 16e3), (16e3, 64e3), (64e3, 256e3), (256e3, 1e6), (1e6, 4e6), (4e6, 1e12)]
RQLAB = ["0", "<1k", "1-4k", "4-16k", "16-64k", "64-256k", "256k-1M", "1-4M", ">4M"]


def regions(C, D, rf, key, W, nev):
    R_ = D[key]
    ev = R_[:, -1].astype(int)
    p, T = R_[:, 0].astype(int), R_[:, 2].astype(np.float64)
    md = R_[:, 4].astype(np.float64)
    di = dbin(md)
    qi = np.full(len(T), -1)
    for i, (lo, hi) in enumerate(RQB[1:]):
        qi[(T > lo) & (T <= hi)] = i
    nq = len(RQB) - 1
    res = {"empty": {}}
    for j, m in enumerate(C["charge"]):
        X = R_[:, 5 + j].astype(np.float64)
        ch = T > 0
        out = {}
        for pp in range(4):
            mp = ((p == pp) if pp < 3 else np.ones(len(p), bool)) & ch
            # linearity: least-squares slope through origin on log-uniform weights? report sum ratio + log-log fit
            lx, ly = np.log10(T[mp]), np.log10(np.maximum(X[mp], 1.0))
            sel = T[mp] > 4e3
            A = np.polyfit(lx[sel], ly[sel], 1) if sel.sum() > 10 else [np.nan, np.nan]
            out[f"{PL[pp]}|fit_loglog_T>4k"] = dict(slope=float(A[0]), intercept=float(A[1]),
                                                    r_log=float(np.corrcoef(lx[sel], ly[sel])[0, 1]))
            for gsel, gname, gl in ((qi, "q", RQLAB[1:]), (di, "d", DLAB)):
                ng = nq if gname == "q" else len(DB)
                g = np.where(mp & (gsel >= 0), gsel, -1)
                Hr = grouped_hist(X / np.maximum(T, 1e-9), g, ev, ng, nev, REDGE)
                med = summarise(Hr, W, REDGE, lambda H: hist_q(H, REDGE, [0.5])[0])
                hw = summarise(Hr, W, REDGE, lambda H: (lambda a: (a[1] - a[0]) / 2)(hist_q(H, REDGE, [0.16, 0.84])))
                # sum ratio per group, bootstrap over events
                S = np.zeros((ng, nev, 2))
                ok = g >= 0
                np.add.at(S, (g[ok], ev[ok], 0), X[ok]); np.add.at(S, (g[ok], ev[ok], 1), T[ok])
                sr_c = S[:, :, 0].sum(1) / np.maximum(S[:, :, 1].sum(1), 1e-9)
                sb = np.einsum("be,gek->gbk", W, S)
                sr_b = sb[..., 0] / np.maximum(sb[..., 1], 1e-9)
                n = Hr.sum((1, 2))
                for gi in range(ng):
                    if n[gi] < 30:
                        continue
                    out[f"{PL[pp]}|{gname}|{gl[gi]}"] = dict(
                        n=int(n[gi]), median_ratio=iv(*med, gi), hw68=iv(*hw, gi),
                        sum_ratio=[float(sr_c[gi]), float(np.percentile(sr_b[gi], 16)), float(np.percentile(sr_b[gi], 84))])
            # density x charge cross for the heatmap
            g = np.where(mp & (di >= 0) & (qi >= 0), di * nq + qi, -1)
            Hr = grouped_hist(X / np.maximum(T, 1e-9), g, ev, len(DB) * nq, nev, REDGE)
            hw = summarise(Hr, W, REDGE, lambda H: (lambda a: (a[1] - a[0]) / 2)(hist_q(H, REDGE, [0.16, 0.84])))
            med = summarise(Hr, W, REDGE, lambda H: hist_q(H, REDGE, [0.5])[0])
            n = Hr.sum((1, 2))
            for gi in range(len(DB) * nq):
                if n[gi] >= 30:
                    out[f"{PL[pp]}|dq|{DLAB[gi // nq]}|{RQLAB[1 + gi % nq]}"] = dict(n=int(n[gi]), median_ratio=iv(*med, gi), hw68=iv(*hw, gi))
        res[m] = out
        for pp in range(4):
            F = rf[:, j, pp] if pp < 3 else rf[:, j].sum(1)          # (nev, 5): n, >0.25Q0, >1Q0, >4Q0, sum
            cen = F.sum(0)
            res["empty"][f"{m}|{PL[pp]}"] = dict(n_empty=int(cen[0]), frac_gt025Q0=float(cen[1] / cen[0]),
                                                 frac_gt1Q0=float(cen[2] / cen[0]), frac_gt4Q0=float(cen[3] / cen[0]),
                                                 mean_pred=float(cen[4] / cen[0]))
    return res


# ---------------------------------------------------------------- 5. particles
EB = [(0.1, 0.3), (0.3, 1), (1, 3), (3, 10), (10, 30), (30, 100), (100, 1e9)]
EBL = ["0.1-0.3", "0.3-1", "1-3", "3-10", "10-30", "30-100", ">100"]
OB = [(-0.01, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]
OBL = ["<0.1", "0.1-0.3", "0.3-0.6", ">0.6"]


def particles(C, D, W, nev):
    P = D["par"]
    ev = P[:, -1].astype(int)
    p = col(C, D, "par", "plane").astype(int)
    E, mixed, Tm = col(C, D, "par", "E"), col(C, D, "par", "mixed_frac"), col(C, D, "par", "T_maj")
    own, ownm = col(C, D, "par", "own"), col(C, D, "par", "own_maj")
    ei = np.full(len(E), -1); oi = np.full(len(E), -1)
    for i, (lo, hi) in enumerate(EB):
        ei[(E > lo) & (E <= hi)] = i
    for i, (lo, hi) in enumerate(OB):
        oi[(mixed > lo) & (mixed <= hi)] = i
    res = {"purity_own_maj_over_own": float(np.median(ownm / own)), "n_particles": int(len(E)),
           "n_by_E": {EBL[i]: int((ei == i).sum()) for i in range(len(EB))}}
    for m in C["charge"]:
        X = col(C, D, "par", m)
        r = X / np.maximum(Tm, 1e-9)
        out = {}
        for pp in range(4):
            mp = (p == pp) if pp < 3 else np.ones(len(p), bool)
            for gsel, nm, labs in ((ei, "E", EBL), (oi, "overlap", OBL)):
                ng = len(labs)
                g = np.where(mp & (gsel >= 0) & (E > 0.1), gsel, -1)
                Hr = grouped_hist(r, g, ev, ng, nev, REDGE)
                med = summarise(Hr, W, REDGE, lambda H: hist_q(H, REDGE, [0.5])[0])
                hw = summarise(Hr, W, REDGE, lambda H: (lambda a: (a[1] - a[0]) / 2)(hist_q(H, REDGE, [0.16, 0.84])))
                n = Hr.sum((1, 2))
                for gi in range(ng):
                    if n[gi] >= 30:
                        out[f"{PL[pp]}|{nm}|{labs[gi]}"] = dict(n=int(n[gi]), median_ratio=iv(*med, gi), hw68=iv(*hw, gi))
            # energy x overlap
            g = np.where(mp & (ei >= 0) & (oi >= 0) & (E > 0.1), ei * len(OB) + oi, -1)
            Hr = grouped_hist(r, g, ev, len(EB) * len(OB), nev, REDGE)
            med = summarise(Hr, W, REDGE, lambda H: hist_q(H, REDGE, [0.5])[0])
            hw = summarise(Hr, W, REDGE, lambda H: (lambda a: (a[1] - a[0]) / 2)(hist_q(H, REDGE, [0.16, 0.84])))
            n = Hr.sum((1, 2))
            for gi in range(len(EB) * len(OB)):
                if n[gi] >= 30:
                    out[f"{PL[pp]}|EO|{EBL[gi // len(OB)]}|{OBL[gi % len(OB)]}"] = dict(n=int(n[gi]), median_ratio=iv(*med, gi), hw68=iv(*hw, gi))
        res[m] = out
    return res


# ---------------------------------------------------------------- 6. separation
def separation(C, D, W, nev, found=0.25 * Q0):
    S_ = D["sep"]
    ev = S_[:, -1].astype(int)
    p, ax, s = (col(C, D, "sep", k).astype(int) for k in ("plane", "axis", "s"))
    Tp, dv = col(C, D, "sep", "Tpmin"), dbin(col(C, D, "sep", "dens_valley"))
    res = {"counts": {}}
    groups = {}
    for pp in range(4):
        mp = (p == pp) if pp < 3 else np.ones(len(p), bool)
        for a in (0, 1):
            for ss in (2, 3, 4, 0):
                for d in list(range(len(DB))) + [-9]:
                    m = mp & (ax == a) & ((s == ss) if ss else True) & ((dv == d) if d != -9 else True)
                    k = f"{PL[pp]}|{'wire' if a == 0 else 'time'}|s{ss if ss else 'all'}|{DLAB[d] if d != -9 else 'all'}"
                    if m.sum() >= 30:
                        groups[k] = m
    for k, m in groups.items():
        res["counts"][k] = int(m.sum())
    for mth in C["charge"]:
        P_, V_ = col(C, D, "sep", f"{mth}_pmin"), col(C, D, "sep", f"{mth}_vmin")
        f_ = (P_ > found).astype(float)
        r_ = (f_ > 0) & (V_ < 0.5 * P_)
        out = {}
        for k, m in groups.items():
            e = ev[m]
            nf = np.bincount(e, weights=f_[m], minlength=nev); nr = np.bincount(e, weights=r_[m], minlength=nev)
            n = np.bincount(e, minlength=nev).astype(float)
            bf, br, bn = W @ nf, W @ nr, W @ n
            fd = bf / np.maximum(bn, 1); rf = br / np.maximum(bf, 1); ru = br / np.maximum(bn, 1)
            out[k] = dict(both_found=[float(nf.sum() / n.sum()), float(np.percentile(fd, 16)), float(np.percentile(fd, 84))],
                          resolved_given_found=[float(nr.sum() / max(nf.sum(), 1)), float(np.percentile(rf, 16)), float(np.percentile(rf, 84))],
                          resolved=[float(nr.sum() / n.sum()), float(np.percentile(ru, 16)), float(np.percentile(ru, 84))],
                          median_dip_found=float(np.median((V_[m] / np.maximum(P_[m], 1e-9))[f_[m] > 0])) if (f_[m] > 0).any() else None)
        res[mth] = out
    return res


# ---------------------------------------------------------------- 7. event level
def event_level(C, D, W, nev):
    E_ = D["evt"]
    cols = C["evt_cols"]
    ev = E_[:, -1].astype(int)
    p = E_[:, 0].astype(int)
    T = E_[:, 1]
    res = {}
    for m in C["charge"]:
        Xs, r, rl, rf = (E_[:, cols.index(f"{m}_{x}")] for x in ("sum", "r", "rlog", "rlog_foot"))
        for pp in range(4):
            mp = (p == pp) if pp < 3 else np.ones(len(p), bool)
            ratio = Xs[mp] / T[mp]
            sT = np.bincount(ev[mp], weights=T[mp], minlength=nev); sX = np.bincount(ev[mp], weights=Xs[mp], minlength=nev)
            bt = (W @ sX) / (W @ sT)
            res[f"{m}|{PL[pp]}"] = dict(
                total_ratio=[float(sX.sum() / sT.sum()), float(np.percentile(bt, 16)), float(np.percentile(bt, 84))],
                per_plane_ratio_median=float(np.median(ratio)), per_plane_ratio_p16_p84=[float(x) for x in np.percentile(ratio, [16, 84])],
                corr_lin_median=float(np.nanmedian(r[mp])), corr_log_median=float(np.nanmedian(rl[mp])),
                corr_log_footprint_median=float(np.nanmedian(rf[mp])),
                corr_log_footprint_p16=float(np.nanpercentile(rf[mp], 16)))
    return res


def coverage(C, D):
    sig = D["sig"]
    p, cov, di = col(C, D, "sig", "plane").astype(int), col(C, D, "sig", "cov"), dbin(col(C, D, "sig", "dens_q"))
    T = col(C, D, "sig", "T")
    res = {}
    for pp in range(4):
        mp = (p == pp) if pp < 3 else np.ones(len(p), bool)
        for a in range(len(DB)):
            s = mp & (di == a)
            if s.sum():
                res[f"{PL[pp]}|{DLAB[a]}"] = dict(n=int(s.sum()), covered=float(cov[s].mean()),
                                                  charge_covered=float((T[s] * cov[s]).sum() / T[s].sum()))
    emp = D["emp"]
    pe, ce, de, we = (col(C, D, "emp", k) for k in ("plane", "cov", "dens_q", "w"))
    dd = dbin(de)
    for a in range(len(DB)):
        s = dd == a
        if s.sum():
            res[f"empty|all|{DLAB[a]}"] = dict(covered=float((ce[s] * we[s]).sum() / we[s].sum()))
    return res


def zeroed(C, D):
    """Share of the true charge (and of charged cells) that each charge output sets to exactly 0, by density."""
    p, T, di = col(C, D, "sig", "plane").astype(int), col(C, D, "sig", "T"), dbin(col(C, D, "sig", "dens_q"))
    qi = qbin(T)
    res = {}
    for m in C["charge"]:
        z = col(C, D, "sig", m) <= 0
        for pp in range(4):
            mp = (p == pp) if pp < 3 else np.ones(len(p), bool)
            for a in list(range(len(DB))) + [-9]:
                s = mp & ((di == a) if a != -9 else True)
                if s.sum():
                    res[f"{m}|{PL[pp]}|{DLAB[a] if a != -9 else 'all'}"] = dict(
                        charge_frac=float(T[s & z].sum() / T[s].sum()), cell_frac=float(z[s].mean()))
                for b in range(len(QB)):                      # density x cell charge
                    sb = s & (qi == b)
                    if sb.sum() >= 30:
                        res[f"{m}|{PL[pp]}|{DLAB[a] if a != -9 else 'all'}|{QLAB[b]}"] = dict(
                            n=int(sb.sum()), charge_frac=float(T[sb & z].sum() / T[sb].sum()), cell_frac=float(z[sb].mean()),
                            charge_zeroed=float(T[sb & z].sum()))
            res[f"{m}|{PL[pp]}|total_charge_zeroed"] = float(T[mp & z].sum())
            res[f"{m}|{PL[pp]}|total_charge"] = float(T[mp].sum())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    C, D, fc, rf16, rf4, nev = load(a.rec)
    W = boot_weights(nev)
    only = set(a.only.split(",")) if a.only else None
    res = dict(meta=dict(n_events=nev, B=B, Q0=Q0, density_bins=dict(zip(DLAB, DB)), charge_bins=dict(zip(QLAB, QB)),
                         rec=a.rec, n_sig=int(len(D["sig"])), n_emp_sampled=int(len(D["emp"]))))
    steps = [("coverage", lambda: coverage(C, D)),
             ("zeroed", lambda: zeroed(C, D)),
             ("fidelity", lambda: fidelity(C, D, W, nev)),
             ("false_charge", lambda: false_charge(C, fc, W)),
             ("false_charge_q", lambda: false_charge_quantiles(C, D)),
             ("regions16x8", lambda: regions(C, D, rf16, "reg16", W, nev)),
             ("regions4x4", lambda: regions(C, D, rf4, "reg4", W, nev)),
             ("particles", lambda: particles(C, D, W, nev)),
             ("separation", lambda: separation(C, D, W, nev)),
             ("event", lambda: event_level(C, D, W, nev)),
             ("presence", lambda: presence(C, D, W, nev))]
    if os.path.exists(a.out) and only:
        res = json.load(open(a.out))
    for name, fn in steps:
        if only and name not in only:
            continue
        import time
        t0 = time.time()
        res[name] = fn()
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(a.out, "w"), indent=0)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
