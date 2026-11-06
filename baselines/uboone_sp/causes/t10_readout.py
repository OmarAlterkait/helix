#!/usr/bin/env python3
"""T10: where in M7dc is bright-charge precision lost? Linear (ridge) readouts of log T from each stage, fitted on
calibration events (probe-train ev000-063 minus 41; never trained on), lambda chosen on the held-out part of them,
scored on the 48 floor test events, per plane and truth bin. Cells: training-label complete truth T >= 16 k e-.
Everything in fp32 (no bf16 rounding in the features).

Stages (the decoder = helix.model.denoise.CellDecoder, reimplemented here to capture its states):
  y        the model's charge output, affine recalibration (baseline)
  rawin    the decoder's direct input: aux + covering tokens' (arcsinh value, occupancy) slots, 4 bands
  sinhin   the same slots as amplitudes sinh(value) (linear in the measured charge) + aux
  enc      encoder features (before the head): covering token per band + mean over its 3x3 neighbourhood, 4 bands
  znbr     z = kv(norm(feats)) of all 3x3 neighbourhood tokens per band: exactly what the decoder's attention reads
  q0       decoder query after its input MLP (before attention)
  blk<i>   decoder state after attention block i (the last one feeds the linear output layer)
Reading: the first stage whose readout error rises to the model's (~0.035 at >= 64 k) is where precision is lost.
T1f already shows the raw coefficients of the 3x3 token neighbourhood carry it (exact inversion 0.018 / 0.008, Y).

    CUDA_VISIBLE_DEVICES=0 scripts/helix_run.sh python baselines/uboone_sp/causes/t10_readout.py --out <json>
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
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
CKPT = f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt"
Q0 = 2043.2578108909702
BINS = [(16e3, 64e3), (64e3, 256e3), (256e3, np.inf)]
LAMS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", default="0-63")
    ap.add_argument("--holdout-from", type=int, default=50, help="cal events >= this select lambda")
    ap.add_argument("--test", default="260-307")
    ap.add_argument("--tmin", type=float, default=16e3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.model.tokenize import assemble, to_fm
    from helix.probe import resolution as R_
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta["head_kw"])
    model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
    head = model.head
    ds = DenoiseEvents(CORPUS, [], CT, items=[]); cfg = ds.cfg

    def item(ev):
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        shard = str(z["shard"]); event = int(z["event"]); run = os.path.basename(os.path.dirname(shard))
        path = os.path.join(CORPUS, run, os.path.basename(shard))
        if not os.path.exists(path):
            return None
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
            c = {k: fh["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        return run, path, pos, event, c

    @torch.no_grad()
    def features(ev):
        it = item(ev)
        if it is None:
            return None
        run, path, pos, event, c = it
        tkeys, tq, tev = ds._truth(run, path, pos)
        assert tev == event
        sel = tq >= a.tmin
        keys, T = tkeys[sel].astype(np.int64), tq[sel].astype(np.float64)
        ce = read_coeff_event(path, pos)
        B = to_fm(assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, gids=c["gids"], n_wires=c["n_wires"],
                           band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg))
        idx, aux = R_.cell_inputs(keys, B["cell_key"], c["band_lengths"], cfg)
        ok = (idx >= 0).any(1)
        nbr = R_.cell_neighbors(keys, B["cell_key"], c["band_lengths"], cfg)
        keys, T, idx, aux, nbr = keys[ok], T[ok], idx[ok], aux[ok], nbr[ok]
        g, _, _ = R_.unkey(keys)
        Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        feats = model.fm.encode(Bt).float()                                   # (N_tok, d)
        out = {k: [] for k in ("y", "rawin", "sinhin", "enc", "znbr", "q0") + tuple(f"blk{i}" for i in range(len(head.blocks)))}
        for s in range(0, len(keys), 8192):
            ti = torch.as_tensor(idx[s:s + 8192]).to(dev); ta = torch.as_tensor(aux[s:s + 8192]).to(dev).float()
            tn = torch.as_tensor(nbr[s:s + 8192]).to(dev)
            n = ti.shape[0]
            # --- CellDecoder.forward, with its states kept ---
            z = head.kv(head.norm(feats))
            flat = tn.reshape(n, -1)
            nt = z.shape[0]
            spread = torch.arange(flat.numel(), device=dev).view_as(flat) % nt
            gidx = torch.where(flat >= 0, flat, spread)
            keep = torch.cat([flat >= 0, torch.ones(n, 1, dtype=torch.bool, device=dev)], 1)
            raw = torch.cat([Bt["inp"], Bt["occ"].to(Bt["inp"].dtype)], -1).float()
            r = raw[ti.clamp(min=0)] * (ti >= 0).unsqueeze(-1).float()       # (n, nb, 2*n_slot)
            q = head.q0(torch.cat([ta, r.flatten(1)], 1)).float()
            out["q0"].append(q.clone())
            h, P = head.heads, q.shape[-1]
            L = flat.shape[1]
            for bi, blk in enumerate(head.blocks):
                zn, nn_ = blk["nk"](z), blk["nk"](head.null)
                Kt, Vt = blk["k"](zn).float(), blk["v"](zn).float()
                kp, vp = blk["k"](head.pos).float(), blk["v"](head.pos).float()
                K = torch.cat([Kt[gidx] + kp[:L], (blk["k"](nn_).float() + kp[L]).expand(n, 1, P)], 1)
                V = torch.cat([Vt[gidx] + vp[:L], (blk["v"](nn_).float() + vp[L]).expand(n, 1, P)], 1)
                Qh = blk["q"](blk["nq"](q)).float().view(n, h, P // h)
                sc = torch.einsum("nhd,nlhd->nhl", Qh, K.view(n, L + 1, h, P // h)) * (P // h) ** -0.5
                att = torch.einsum("nhl,nlhd->nhd", sc.masked_fill(~keep[:, None, :], float("-inf")).softmax(-1),
                                   V.view(n, L + 1, h, P // h)).reshape(n, P)
                q = q + blk["o"](att).float()
                q = q + blk["mlp"](blk["nm"](q)).float()
                out[f"blk{bi}"].append(q.clone())
            y = head.out(q)[:, 0]
            out["y"].append(y[:, None])
            nb = r.shape[1]; ns = r.shape[2] // 2
            out["rawin"].append(torch.cat([ta, r.flatten(1)], 1))
            out["sinhin"].append(torch.cat([ta, torch.sinh(r[:, :, :ns]).flatten(1)], 1))
            fc = feats[ti.clamp(min=0)] * (ti >= 0).unsqueeze(-1).float()      # (n, nb, d)
            fn = feats[gidx].view(n, nb, -1, feats.shape[1]) * (tn >= 0).unsqueeze(-1).float()
            fm = fn.sum(2) / (tn >= 0).sum(2, keepdim=True).clamp(min=1)
            out["enc"].append(torch.cat([fc.flatten(1), fm.flatten(1)], 1))
            out["znbr"].append((z[gidx] * (flat >= 0).unsqueeze(-1).float()).flatten(1))
        X = {k: torch.cat(v).double() for k, v in out.items()}
        return X, torch.as_tensor(np.log(T), device=dev), torch.as_tensor(g % 3, device=dev), T

    cal_lo, cal_hi = map(int, a.cal.split("-"))
    cal = [e for e in range(cal_lo, cal_hi + 1) if e != 41]
    fitA = [e for e in cal if e < a.holdout_from]; fitB = [e for e in cal if e >= a.holdout_from]
    names = None
    acc = {}                                # (fold, set, plane) -> [XtX, Xty, n]

    def accum(fold, evs):
        nonlocal names
        for ev in evs:
            t0 = time.time()
            F = features(ev)
            if F is None:
                print(f"ev{ev:03d} skipped", flush=True); continue
            X, yt, pl, _ = F
            names = list(X)
            for k, x in X.items():
                xa = torch.cat([x, torch.ones(len(x), 1, device=dev, dtype=x.dtype)], 1)
                for p in range(3):
                    m = pl == p
                    key = (fold, k, p)
                    xm, ym = xa[m], yt[m]
                    if key not in acc:
                        acc[key] = [torch.zeros(xa.shape[1], xa.shape[1], device=dev, dtype=torch.float64),
                                    torch.zeros(xa.shape[1], device=dev, dtype=torch.float64), 0]
                    acc[key][0] += xm.T @ xm; acc[key][1] += xm.T @ ym; acc[key][2] += int(m.sum())
            print(f"{fold} ev{ev:03d} {time.time() - t0:.1f}s", flush=True)

    def solve(XtX, Xty, lam):
        """Ridge on standardised features from the accumulated normal equations (last column = intercept)."""
        n = XtX[-1, -1]
        mu = XtX[:-1, -1] / n; ybar = Xty[-1] / n
        C = XtX[:-1, :-1] / n - torch.outer(mu, mu)
        sd = torch.sqrt(torch.diagonal(C).clamp(min=1e-20))
        Cs = C / torch.outer(sd, sd)
        cy = (Xty[:-1] / n - mu * ybar) / sd
        ws = torch.linalg.solve(Cs + lam * torch.eye(len(sd), device=dev, dtype=Cs.dtype), cy)
        w = ws / sd
        return torch.cat([w, (ybar - mu @ w)[None]])

    def score(evs, W):                       # W[(set, plane)] -> weights; returns residuals per (set, plane, bin)
        res = {}
        for ev in evs:
            F = features(ev)
            if F is None:
                continue
            X, yt, pl, T = F
            Tt = torch.as_tensor(T, device=dev)
            for k, x in X.items():
                xa = torch.cat([x, torch.ones(len(x), 1, device=dev, dtype=x.dtype)], 1)
                for p in range(3):
                    m = pl == p
                    for lam_key, w in W[(k, p)].items():
                        e = (xa[m] @ w - yt[m])
                        for lo, hi in BINS:
                            mb = (Tt[m] >= lo) & (Tt[m] < hi)
                            res.setdefault((k, p, lam_key, lo), []).append(e[mb].cpu().numpy())
        return {kk: np.concatenate(v) for kk, v in res.items()}

    def hw(e):
        if len(e) < 20:
            return float("nan")
        a16, a84 = np.percentile(e, [16, 84]); return float((a84 - a16) / 2)

    accum("A", fitA)
    # lambda choice on fold B (mean hw68 over the three bins, per stage and plane)
    W = {(k, p): {lam: solve(acc[("A", k, p)][0], acc[("A", k, p)][1], lam) for lam in LAMS} for k in names for p in range(3)}
    rb = score(fitB, W)
    best = {}
    for k in names:
        for p in range(3):
            sc = {lam: np.nanmean([hw(rb[(k, p, lam, lo)]) for lo, _ in BINS]) for lam in LAMS}
            best[(k, p)] = min(sc, key=sc.get)
    accum("B", fitB)
    Wf = {}
    for k in names:
        for p in range(3):
            XtX = acc[("A", k, p)][0] + acc[("B", k, p)][0]; Xty = acc[("A", k, p)][1] + acc[("B", k, p)][1]
            Wf[(k, p)] = {"best": solve(XtX, Xty, best[(k, p)])}
    lo_t, hi_t = map(int, a.test.split("-"))
    rt = score(list(range(lo_t, hi_t + 1)), Wf)
    out = {"cal_events": cal, "lambda": {f"{k}_{'UVY'[p]}": best[(k, p)] for k in names for p in range(3)}, "test": {}}
    for k in names:
        for p in range(3):
            out["test"][f"{k}_{'UVY'[p]}"] = {f"{lo:.0f}": dict(hw68=hw(rt[(k, p, 'best', lo)]), med=float(np.median(rt[(k, p, 'best', lo)])),
                                                              n=int(len(rt[(k, p, 'best', lo)]))) for lo, _ in BINS}
    json.dump(out, open(a.out, "w"), indent=1)
    for p in range(3):
        print("UVY"[p], "  ".join(k + " " + "/".join(f"{out['test'][k + '_' + 'UVY'[p]][f'{lo:.0f}']['hw68']:.4f}" for lo, _ in BINS)
                                  for k in names), flush=True)

if __name__ == "__main__":
    main()
