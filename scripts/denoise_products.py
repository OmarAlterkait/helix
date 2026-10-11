#!/usr/bin/env python3
"""Calibrated products of a trained denoiser (post hoc; no retraining).

The denoiser is trained on CASE-CONTROL rows (helix.data.denoise.DenoiseEvents.rows: every
'charged' cell up to sig_cap plus empty cells drawn by label-dependent rules), with MSE on
y = log1p(q/Q0) and BCE on presence = (training truth q > support_min_q). Its outputs are
therefore posterior quantities under the TRAINING distribution
(baselines/uboone_sp/dense/results/firstp.md): presence is over-confident by the case-control
odds ratio (9-10x on average, ~15x in sparse surroundings, ~1.5x in shower cores), and
expm1 of a log-space mean under-estimates E[q] (Jensen). This module fits, on CALIBRATION
events (never the test events), and applies:

  (a) prior correction   logit' = logit - log OR[plane][pdens],  OR = odds(charged | training
                         rows) / odds(charged | all token-covered cells), per plane and
                         stratum of the model's own predicted charge within +-8 fine cells
                         excluding the cell ('pdens': deployable, uses no truth)
  (b) Jensen table       sigma^2[ybin][pdens] of the log-space residual (training-truth y -
                         y_hat): over all covered cells (non-hurdle head) or over training-
                         charged cells (hurdle head: the conditional charge)
  (c) products, per cell:
        p_cal   calibrated presence = sigmoid(logit - log OR)
        calo    calorimetry map E[q | data]:
                  hurdle      p_cal * Q0 * (exp(y + sigma^2/2) - 1)
                  non-hurdle  Q0 * (exp(y + sigma^2/2) - 1) * p_cal / p      (prior shift + Jensen;
                              firstp: the most nearly unbiased in linear space above ~1 k e-)
        clean   the decision map: calo where p_cal > tau[plane], else 0. tau minimises, on the
                calibration events, the L1 charge error of the clean map against the COMPLETE
                truth over all readout cells: sum_{p_cal <= tau} q + sum_{p_cal > tau} |calo - q|
                = zeroed charge + mis-measured (incl. false) charge.

Fit (calibration events = probe-train ev000-063 minus ev041 by default; dumps from
baselines/uboone_sp/dense/model_dump2.py, i.e. charge ``q_<tag>_g<g>`` = Q0*expm1(max(y,0)),
presence ``p_<tag>_g<g>``, coverage ``cov_g<g>``):

    scripts/helix_run.sh python scripts/denoise_products.py fit --model M6d:<ckpt>:<dump dir>[:<map tag>] \
        [--model ...] --out PARAMS.json

Apply to dumps (writes p_cal / calo / clean maps per plane):

    scripts/helix_run.sh python scripts/denoise_products.py apply --params PARAMS.json --tag M6d \
        --dump <dump dir>/ev300.npz --out products_ev300.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Q0_DEFAULT = 2043.2578108909702
DEDGE = np.array([-1, 0.5, 1e4, 1e5, 1e6, 4e6, np.inf]); ND = len(DEDGE) - 1
DLAB = ["iso", "<=10k", "10k-100k", "100k-1M", "1M-4M", ">4M"]
YEDGE = np.array([-np.inf, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, np.inf]); NY = len(YEDGE) - 1
TAUS = np.round(np.arange(0.01, 1.0, 0.01), 2)
TRUTH = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
PATH_MAP = [("/pscratch/sd/o/oalter/coeff_tpc_r1", "/global/cfs/cdirs/m5238/users/oalter/coeff_tpc_r1"),
            ("/pscratch/sd/o/oalter/denoise_truth", "/global/cfs/cdirs/m5238/users/oalter/ost61/denoise_truth")]
COMPLETE = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
CAL_EVENTS = [e for e in range(0, 64) if e != 41]


def remap(p):
    for a, b in PATH_MAP:
        if p.startswith(a):
            return b + p[len(a):]
    return p


# ----------------------------------------------------------------------------------------- products
def box_excl(Q, r=8):
    """Sum of Q over the (2r+1)^2 box around each cell, the cell itself excluded."""
    X = np.asarray(Q, np.float64)
    P = np.pad(X, ((r + 1, r), (r + 1, r)))
    S = P.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k] - X


def dbin(x):
    return np.clip(np.searchsorted(DEDGE, x, side="left") - 1, 0, ND - 1)


def ybin(y):
    return np.clip(np.searchsorted(YEDGE, y, side="right") - 1, 0, NY - 1)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def products(y, p, plane, pdens, prm, cov=None):
    """Products for cells with raw charge-head output ``y`` (log1p(q/Q0) units), presence
    probability ``p``, plane index (0/1/2 = U/V/Y; scalar or array) and predicted density
    ``pdens`` (e-, box sum of the model's own charge, cell excluded). Returns dict of arrays."""
    Q0 = prm["q0"]
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    pl = np.broadcast_to(np.asarray(plane), y.shape)
    db = dbin(np.asarray(pdens, np.float64))
    lor = np.asarray(prm["log_or"])[pl, db]
    p_cal = 1.0 / (1.0 + np.exp(-(logit(p) - lor)))
    yc = np.maximum(y, 0.0)
    s2 = np.asarray(prm["sigma2"])[ybin(yc), db]
    qJ = Q0 * np.expm1(yc + s2 / 2) * (yc > 0)
    if prm["hurdle"]:
        calo = p_cal * qJ
    else:
        calo = qJ * p_cal / np.maximum(p, 1e-7)
    tau = np.asarray(prm["tau"])[pl]
    clean = np.where(p_cal > tau, calo, 0.0)
    if cov is not None:
        p_cal, calo, clean = (np.where(cov, x, 0.0) for x in (p_cal, calo, clean))
    return dict(p_cal=p_cal, calo=calo, clean=clean)


def products_maps(qmap, pmap, cov, plane, prm):
    """Products on a full fine-cell plane map (charge map q = Q0*expm1(max(y,0)), presence p)."""
    Q0 = prm["q0"]
    y = np.log1p(np.maximum(qmap, 0) / Q0)
    return products(y, pmap, plane, box_excl(qmap), prm, cov=cov)


# ----------------------------------------------------------------------------------------- fit
def _shapes():
    from helix.tpc.geometry import load_plane_registry
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    return {g: (-(-reg[g]["n_wires"] // 2), -(-reg[g]["n_ticks"] // 16)) for g in range(6)}


def _corpus_truth(root, shard, event):
    run = os.path.basename(os.path.dirname(shard)); tag = os.path.basename(shard).replace("sim_wire_coeff_", "").replace(".h5", "")
    z = np.load(os.path.join(root, run, f"truth_{tag}.npz"))
    i = int(np.nonzero(z["event"] == event)[0][0])
    a, b = z["offsets"][i], z["offsets"][i + 1]
    return z["keys"][a:b].astype(np.int64), z["q"][a:b].astype(np.float64)


def _maps(keys, q, shapes):
    from helix.probe.resolution import unkey
    g, w, t = unkey(keys)
    out = {}
    for gg, sh in shapes.items():
        M = np.zeros(sh)
        m = (g == gg) & (w < sh[0]) & (t >= 0) & (t < sh[1])
        np.add.at(M, (w[m], t[m]), q[m])
        out[gg] = M
    return out


def _dump_planes(Mz, name, g, shape):
    to = int(Mz["toff"])
    crop = lambda X: X[:shape[0], to:to + shape[1]].astype(np.float64)
    return crop(Mz[f"q_{name}_g{g}"]), crop(Mz[f"p_{name}_g{g}"]), crop(Mz[f"cov_g{g}"]) > 0


def _event_inputs(ev, spec):
    import h5py
    z = np.load(os.path.join(TRUTH, f"ev{ev:03d}.npz"), allow_pickle=True)
    shard, event = remap(str(z["shard"])), int(z["event"])
    with h5py.File(shard, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], event))
    return shard, event, pos


def _fit_pass1(job):
    """Per calibration event: deployment / training-row counts and residual sums."""
    ev, spec = job
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.probe.resolution import unkey
    shapes = _shapes()
    shard, event, pos = _event_inputs(ev, spec)
    tk, tq = _corpus_truth(spec["truth_root"], shard, event)
    o = np.argsort(tk); tk, tq = tk[o], tq[o]
    ds = DenoiseEvents("", [], "", items=[], **spec["recipe"])
    rows, _ = ds.rows(read_coeff_event(shard, pos), tk, tq.astype(np.float32), np.random.default_rng(spec["seed0"] + ev))
    rg, rw, rt = unkey(rows)
    T = _maps(tk, tq, shapes)
    Mz = np.load(os.path.join(spec["dump"], f"ev{ev:03d}.npz"))
    C = np.zeros((3, ND, 4)); RT = np.zeros((NY, ND, 3))
    Q0, c = spec["q0"], spec["support_min_q"]
    for g, sh in shapes.items():
        q, p, cov = _dump_planes(Mz, spec["map"], g, sh)
        pd = dbin(box_excl(q))
        lab = T[g] > c
        R = np.zeros(sh, bool)
        m = (rg == g) & (rw < sh[0]) & (rt >= 0) & (rt < sh[1])
        R[rw[m], rt[m]] = True
        R &= cov
        for sel, j in ((cov, 0), (R, 2)):
            C[g % 3, :, j] += np.bincount(pd[sel], minlength=ND)
            C[g % 3, :, j + 1] += np.bincount(pd[sel], weights=lab[sel].astype(float), minlength=ND)
        yh = np.log1p(q / Q0); yt = np.log1p(T[g] / Q0)
        sel = cov & lab if spec["hurdle"] else cov
        idx = ybin(yh[sel]) * ND + pd[sel]
        r = yt[sel] - yh[sel]
        for j, w in enumerate((None, r, r * r)):
            RT[:, :, j] += np.bincount(idx, weights=w, minlength=NY * ND).reshape(NY, ND)
    return C, RT


def _fit_pass2(job):
    """Per calibration event: the clean-map cost curve per plane over TAUS (complete truth)."""
    ev, spec, prm = job
    shapes = _shapes()
    shard, event, pos = _event_inputs(ev, spec)
    ck, cq = _corpus_truth(COMPLETE, shard, event)
    Tc = _maps(ck, cq, shapes)
    Mz = np.load(os.path.join(spec["dump"], f"ev{ev:03d}.npz"))
    cost = np.zeros((3, len(TAUS)))
    edges = np.concatenate([[-1.0], TAUS, [2.0]])
    for g, sh in shapes.items():
        q, p, cov = _dump_planes(Mz, spec["map"], g, sh)
        P = products_maps(q, p, cov, g % 3, dict(prm, tau=[2.0] * 3))     # tau irrelevant here
        pc, calo, T = P["p_cal"].ravel(), P["calo"].ravel(), Tc[g].ravel()
        b = np.clip(np.searchsorted(edges, pc, side="left") - 1, 0, len(edges) - 2)   # bin k: TAUS[k-1] < p <= TAUS[k]
        drop = np.bincount(b, weights=T, minlength=len(edges) - 1)
        keep = np.bincount(b, weights=np.abs(calo - T), minlength=len(edges) - 1)
        # tau = TAUS[k]: cells with p_cal <= TAUS[k] dropped (bins 0..k), the rest kept
        cd, ck_ = np.cumsum(drop), np.cumsum(keep[::-1])[::-1]
        cost[g % 3] += cd[:len(TAUS)] + ck_[1:len(TAUS) + 1]
    return cost


def model_spec(name, ckpt, dump, mapname):
    import torch
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)["meta"]
    a = meta["args"]
    recipe = dict(sig_cap=a.get("sig_cap") or 20000, neg_per_sig=a.get("neg_per_sig") or 1.0,
                  near_per_sig=a.get("near_per_sig") or 1.0, cov_per_sig=a.get("cov_per_sig", 1.0),
                  win_per_event=a.get("win_per_event", 0), near_any_per_sig=a.get("near_any_per_sig", 0.0),
                  win_near_per_event=a.get("win_near_per_event", 0))
    c = float(a.get("support_min_q") or 0.0)
    if c:
        recipe["support_min_q"] = c
    troot = remap(a["truth_root"])
    return dict(name=name, ckpt=ckpt, dump=dump, map=mapname, truth_root=troot, recipe=recipe,
                seed0=23 if os.path.realpath(troot) == os.path.realpath(COMPLETE) else 11,      # firstp's row seeds
                support_min_q=c, hurdle=bool(meta.get("cond_charge") or a.get("cond_charge")), q0=float(meta["q0"]))


def fit_model(spec, events, workers):
    from multiprocessing import get_context
    with get_context("spawn").Pool(workers, maxtasksperchild=4) as pool:
        res = pool.map(_fit_pass1, [(e, spec) for e in events])
    C = sum(r[0] for r in res); RT = sum(r[1] for r in res)
    lo = lambda pos, n: np.log((pos + 0.5) / (n - pos + 0.5))
    log_or = lo(C[..., 3], C[..., 2]) - lo(C[..., 1], C[..., 0])                    # (plane, pdens)
    n, s, s2 = RT[..., 0], RT[..., 1], RT[..., 2]
    nm, sm, s2m = n.sum(1, keepdims=True), s.sum(1, keepdims=True), s2.sum(1, keepdims=True)
    use = n >= 200                                                                   # sparse cells: the y-bin marginal
    n_, s_, s2_ = np.where(use, n, nm), np.where(use, s, sm), np.where(use, s2, s2m)
    mu = s_ / np.maximum(n_, 1)
    sigma2 = np.maximum(s2_ / np.maximum(n_, 1) - mu ** 2, 0.0)
    prm = dict(q0=spec["q0"], hurdle=spec["hurdle"], support_min_q=spec["support_min_q"], log_or=log_or.tolist(),
               sigma2=sigma2.tolist(), tau=[0.5, 0.5, 0.5])
    with get_context("spawn").Pool(workers, maxtasksperchild=4) as pool:
        cost = sum(pool.map(_fit_pass2, [(e, spec, prm) for e in events]))
    k = cost.argmin(1)
    prm["tau"] = [float(TAUS[i]) for i in k]
    prm["fit"] = dict(events=events, ckpt=spec["ckpt"], dump=spec["dump"], truth_root=spec["truth_root"],
                      recipe=spec["recipe"], counts=C.tolist(),
                      log_or_glob=(lo(C[..., 3].sum(1), C[..., 2].sum(1)) - lo(C[..., 1].sum(1), C[..., 0].sum(1))).tolist(),
                      tau_cost_rel=(cost / cost.min(1, keepdims=True)).round(4).tolist(), taus=TAUS.tolist(),
                      tau_cost_at_0p5=(cost[:, list(TAUS).index(0.5)] / cost.min(1)).round(4).tolist(),
                      sigma_log_by_ybin=np.sqrt(sigma2[:, 2:4].mean(1)).tolist(),      # pdens 10k-1M, for reading
                      mu_log=mu.tolist(),       # mean log residual per (ybin, pdens): NOT applied (diagnostic)
                      definitions=dict(pdens_edges=DEDGE.tolist(), ybin_edges=YEDGE.tolist(),
                                       tau_cost="sum_{p_cal<=tau} q_complete + sum_{p_cal>tau} |calo - q_complete|, all readout cells"))
    return prm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--model", action="append", required=True, help="NAME:CKPT:DUMPDIR[:MAPTAG]")
    f.add_argument("--events", default="cal", help="'cal' (ev000-063 minus ev041) or lo-hi")
    f.add_argument("--workers", type=int, default=32)
    f.add_argument("--out", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--params", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--dump", required=True)
    p.add_argument("--map", default=None)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "fit":
        events = CAL_EVENTS if a.events == "cal" else list(range(*(int(x) + i for i, x in enumerate(a.events.split("-")))))
        allp = json.load(open(a.out)) if os.path.exists(a.out) else {}
        for spec_s in a.model:
            parts = spec_s.split(":")
            name, ckpt, dump = parts[:3]
            spec = model_spec(name, ckpt, dump, parts[3] if len(parts) > 3 else name)
            prm = fit_model(spec, events, a.workers)
            allp[name] = prm
            json.dump(allp, open(a.out, "w"), indent=1)
            print(name, "tau", prm["tau"], "log OR glob", np.round(prm["fit"]["log_or_glob"], 3), flush=True)
    else:
        prm = json.load(open(a.params))[a.tag]
        Mz = np.load(a.dump)
        shapes = _shapes()
        out = {}
        for g, sh in shapes.items():
            q, pp, cov = _dump_planes(Mz, a.map or a.tag, g, sh)
            for k, v in products_maps(q, pp, cov, g % 3, prm).items():
                out[f"{k}_g{g}"] = v.astype(np.float32)
        np.savez_compressed(a.out, **out)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
