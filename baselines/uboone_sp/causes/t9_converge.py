#!/usr/bin/env python3
"""T9: is M7dc converged, and is its bright-cell error set by convergence, by the objective's trade-off, or by
capacity? Measured at the final weights, on TRAINING events, with the exact training loss (no new run).

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/t9_converge.py --out <json>

Loss per event exactly as scripts/train_denoise.py (hurdle): reg = sum_{pos rows} (p - y)^2 / n_pos,
loss = reg + presence_w * BCE(logit, y > y_sup). reg splits exactly into reg_bright (rows with y >= log1p(64k/Q0))
and reg_rest; g_total = g_bright + g_other (other = reg_rest + BCE).
1. Gradient signal vs noise over N events: G = mean_i g_i, tr(Sigma) = per-event variance; the training batch is
   B = 16 events (one per rank), so SNR_B = |G|^2 / (tr(Sigma) / B), with |G|^2 debiased (|mean|^2 - trSigma/N).
   SNR_B << 1: the optimizer's steps are noise-dominated at this point (converged for this objective and batch);
   SNR_B >~ 1: a consistent descent direction remains. Raw and Adam-preconditioned (u = g / (sqrt(v) + eps), v from
   the checkpoint's optimizer state). Also cos(G_bright, G_other).
   Clipping: E|batch-mean g|^2 = |G|^2 + trSigma / B, against the clip norm 1.0.
2. Line search in fp32 on M held-out training events along the estimated mean direction D (raw gradient, or
   Adam-preconditioned with sqrt(v) floored at its median), step = c * L0 / (G . D) so that the first-order
   predicted decrease is c * L0, c = 0.001 .. 0.3. Along the total direction and the bright-only direction (how far
   bright cells could improve, and what that costs the rest). Noise left in D only adds curvature: a lower bound.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
CKPT = f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt"
Q0 = 2043.2578108909702


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="events for the gradient statistics")
    ap.add_argument("--m", type=int, default=48, help="held-out training events for the line search")
    ap.add_argument("--bright", type=float, default=64e3)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from helix.data.denoise import DenoiseEvents, split_events
    from helix.model.denoise import build_denoise
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    meta = ck["meta"]; args = meta["args"]
    ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta["head_kw"])
    model.load_state_dict(ck["model"]); model.to(dev).eval()
    # parameters in the optimizer's order, with their final lr and Adam second moment
    groups = model.param_groups(args["lr"], args["lr_head"], args["wd"])
    osd = ck["opt"]
    params, lrs, vs = [], [], []
    for g_m, g_s in zip(groups, osd["param_groups"]):
        assert len(g_m["params"]) == len(g_s["params"])
        for p, pid in zip(g_m["params"], g_s["params"]):
            st = osd["state"].get(pid)
            params.append(p); lrs.append(float(g_s["lr"]))
            vs.append(st["exp_avg_sq"].to(dev).float() if st is not None else torch.zeros_like(p, device=dev))
            assert vs[-1].shape == p.shape
    del ck
    eps = 1e-8
    pre = [1.0 / (v.sqrt() + eps) for v in vs]
    print(f"{sum(p.numel() for p in params)/1e6:.1f}M trainable params in {len(params)} tensors; final lr {sorted(set(lrs))}", flush=True)

    runs = meta["runs"]
    items = split_events(CORPUS, runs, "train")
    rng = np.random.default_rng(1)
    pick = rng.permutation(len(items))[:a.n + a.m]
    kw = dict(cov_per_sig=args["cov_per_sig"], win_per_event=args["win_per_event"], near_any_per_sig=args["near_any_per_sig"],
              win_near_per_event=args["win_near_per_event"], neighbors=True, support_min_q=args["support_min_q"],
              sample_seed=4321)
    dsA = DenoiseEvents(CORPUS, runs, CT, items=[items[i] for i in pick[:a.n]], **kw)
    dsB = DenoiseEvents(CORPUS, runs, CT, items=[items[i] for i in pick[a.n:]], **kw)
    y_sup = float(np.log1p(args["support_min_q"] / Q0)); y_br = float(np.log1p(a.bright / Q0))
    pw = float(args.get("presence_w", 1.0))

    def to_dev(it):
        B = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in it["B"].items()}
        B["n_cells"] = B["plane_id"].shape[0]
        return B, torch.as_tensor(it["idx"]).to(dev), torch.as_tensor(it["aux"]).to(dev), torch.as_tensor(it["y"]).to(dev), \
            torch.as_tensor(it["nbr"]).to(dev)

    def losses(it):
        B, idx, aux, y, nbr = to_dev(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, logit = model(B, idx, aux, nbr)
        p, logit = p.float(), logit.float()
        pos = y > y_sup; br = y >= y_br
        npos = pos.sum().clamp(min=1)
        se = (p - y) ** 2
        reg_b = (se * br).sum() / npos
        reg_r = (se * (pos & ~br)).sum() / npos
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, pos.float())
        return reg_b, reg_r, bce, int(br.sum()), int(pos.sum())

    def flat(gs):
        return torch.cat([g.reshape(-1) for g in gs])

    nP = sum(p.numel() for p in params)
    S = {k: torch.zeros(nP, device=dev, dtype=torch.float64) for k in ("b", "o", "ub", "uo")}
    SQ = {k: 0.0 for k in ("b", "o", "t", "ub", "uo", "ut")}
    prec = flat(pre).double()
    nb_rows = npos_rows = 0
    t0 = time.time()
    for i in range(len(dsA)):
        it = dsA[i]
        reg_b, reg_r, bce, nb, npos = losses(it)
        nb_rows += nb; npos_rows += npos
        gb = torch.autograd.grad(reg_b, params, retain_graph=True, allow_unused=True)
        go = torch.autograd.grad(reg_r + pw * bce, params, allow_unused=True)
        gb = flat([torch.zeros_like(p) if g is None else g for g, p in zip(gb, params)]).double()
        go = flat([torch.zeros_like(p) if g is None else g for g, p in zip(go, params)]).double()
        for k, g in (("b", gb), ("o", go)):
            S[k] += g; SQ[k] += float((g * g).sum())
            u = g * prec; S["u" + k] += u; SQ["u" + k] += float((u * u).sum())
        gt = gb + go; SQ["t"] += float((gt * gt).sum()); ut = gt * prec; SQ["ut"] += float((ut * ut).sum())
        if i % 32 == 0:
            print(f"grad {i}/{len(dsA)} {time.time() - t0:.0f}s", flush=True)
    N = len(dsA); Bsz = 16
    res = dict(n_events=N, bright_rows_share=nb_rows / max(npos_rows, 1), y_bright=y_br)

    def stats(mean, sq):
        m2 = float((mean * mean).sum())
        tr = (sq - N * m2) / (N - 1)
        g2 = m2 - tr / N
        return dict(G2=g2, trSigma=tr, SNR_B16=g2 / (tr / Bsz), batch_mean_norm=float(np.sqrt(max(g2, 0) + tr / Bsz)))

    for tag, (kb, ko, kt) in (("raw", ("b", "o", "t")), ("adam", ("ub", "uo", "ut"))):
        Gb, Go = S[kb] / N, S[ko] / N
        res[tag] = dict(bright=stats(Gb, SQ[kb]), other=stats(Go, SQ[ko]), total=stats(Gb + Go, SQ[kt]),
                        cos_bright_other=float((Gb * Go).sum() / ((Gb * Gb).sum().sqrt() * (Go * Go).sum().sqrt())))
    print(json.dumps(res, indent=1), flush=True)

    # Line search (fp32, held-out training events) along the mean directions. Step alpha = c * L0 / (G_hat . D):
    # a first-order predicted decrease of c * L0, c in C. Directions: raw mean gradient, and Adam-preconditioned with
    # sqrt(v) floored at its median (a tiny stored v would otherwise blow single coordinates up). Noise left in the
    # estimated direction only adds curvature, so the measured decrease is a LOWER bound on what descent could gain.
    vflat = torch.cat([v.reshape(-1) for v in vs]).double().sqrt()
    pre_safe = 1.0 / torch.maximum(vflat, vflat.median())
    Gt, Gb = (S["b"] + S["o"]) / N, S["b"] / N
    dirs = {"raw_total": Gt, "raw_bright": Gb, "adam_total": Gt * pre_safe, "adam_bright": Gb * pre_safe}
    theta0 = [p.detach().clone() for p in params]

    def set_theta(step):
        off = 0
        with torch.no_grad():
            for p, t0_ in zip(params, theta0):
                n = p.numel()
                p.copy_(t0_ - step[off:off + n].view_as(p).to(p.dtype)); off += n

    @torch.no_grad()
    def eval_B():
        tot = np.zeros(3)
        for i in range(len(dsB)):
            with torch.autocast("cuda", enabled=False):
                rb, rr, bc, _, _ = losses_fp32(dsB[i])
            tot += [float(rb), float(rr), float(bc)]
        tot /= len(dsB)
        return dict(reg_bright=tot[0], reg_rest=tot[1], bce=tot[2], total=tot[0] + tot[1] + pw * tot[2])

    def losses_fp32(it):
        B, idx, aux, y, nbr = to_dev(it)
        p, logit = model(B, idx, aux, nbr)
        p, logit = p.float(), logit.float()
        pos = y > y_sup; br = y >= y_br
        npos = pos.sum().clamp(min=1)
        se = (p - y) ** 2
        return (se * br).sum() / npos, (se * (pos & ~br)).sum() / npos, \
            torch.nn.functional.binary_cross_entropy_with_logits(logit, pos.float()), 0, 0

    base = eval_B()
    res["line"] = {"base": base}
    print("base", base, flush=True)
    for dname, D in dirs.items():
        ref = "reg_bright" if dname.endswith("bright") else "total"
        L0 = base[ref]
        gd = float((Gb if dname.endswith("bright") else Gt).mul(D).sum())
        res["line"][dname] = {}
        for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
            set_theta(D * (c * L0 / gd))
            r = eval_B(); r["pred_first_order"] = -c * L0
            res["line"][dname][c] = r
            print(dname, c, {k: round(v, 6) for k, v in r.items()}, flush=True)
    set_theta(torch.zeros(nP, device=dev, dtype=torch.float64))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
