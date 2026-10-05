#!/usr/bin/env python3
"""Stage 1 of the resolution evaluation: one checkpoint on the fixed fine grid.

    python scripts/eval_resolution.py --checkpoint <artifact> --truth <dir> --tag T
        --out results.jsonl [--train-events 260] [--boot 200]

Truth comes from ``scripts/dump_resolution_truth.py``. Every model is scored on the
same 2-wire x 16-tick cells whatever its patch size, through FROZEN per-token
encoder features: each cell's input is, per band, the feature of the token that
covers it (zeros and a presence bit if none) plus the cell's continuous offset
inside that token. A bigger patch puts several cells under one feature; the
offset is what lets a probe ask where inside it.

An MLP probe predicts log1p(q / Q0) per cell, trained on the first
``--train-events`` events of the split and scored on the rest. Three arms: the
trained encoder, the same architecture randomly initialised, and the raw token
inputs. Metrics are those of :func:`helix.probe.resolution.scalars`, each with a
[16, 84]% interval from an event bootstrap. Also reported (trained arm):
reconstruction with the SAME physical regions hidden for every model (75% of
32-wire x 128-tick regions), as explained variance and value cross-entropy.

Memory: features are held on the host, ~10-40 GB per process depending on patch
size; several processes on one node can exhaust host memory silently.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import zlib
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="helix eval artifact (scripts/export_artifact.py)")
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="results JSONL, appended")
    ap.add_argument("--train-events", type=int, default=260)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                    help="model-config entries replaced before building (e.g. varlen=False), "
                         "to score the same weights through another code path")
    a = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from helix.core.coeff_io import read_coeff_event
    from helix.model.artifact import build, load
    from helix.model.loss import bucketize_bins
    from helix.model.tokenize import PatchConfig, assemble, pixel_cells, to_fm
    from helix.probe import resolution as R

    dev = torch.device("cuda")
    art = load(a.checkpoint)
    if a.override:
        import ast
        arch = dict(art.arch)
        for kv in a.override:
            k, v = kv.split("=", 1)
            try:
                arch[k] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                arch[k] = v
        art = replace(art, arch=arch)
        print(f"[{a.tag}] overrides: {a.override}", flush=True)
    pw, pt = art.op.pw or 16, art.op.pt or 8
    cfg = PatchConfig(cell_t="grid_center", pw=pw, pt=pt, n_bands=art.op.n_bands or 4)
    nb = cfg.n_bands
    torch.manual_seed(0)
    models = {"trained": build(art, device=dev, eval_mode=True)}
    torch.manual_seed(0)                                        # the random-init control, seeded
    models["random"] = build(replace(art, state_dict=None), device=dev, eval_mode=True)
    print(f"[{a.tag}] pw={pw} pt={pt} bands={nb} d={models['trained'].d}", flush=True)

    def fourier(x):
        ang = 2 * np.pi * x[..., None] * np.array([1, 2, 3], np.float32)
        return np.concatenate([np.sin(ang), np.cos(ang)], -1).reshape(x.shape[0], -1)

    def tokens(shard, event):
        with h5py.File(shard, "r") as f:
            ids = f["ident"]["event"][:]
            gids, nw, bl, ns = (f["config"][x][:] for x in ("gids", "n_wires", "band_lengths", "norm_sigma"))
        pos = int(np.searchsorted(ids, event))
        ce = read_coeff_event(shard, pos)
        cl = read_coeff_event(shard.replace("sim_wire_coeff_", "sim_wire_coeff_clean_"), pos, coords_from=ce)
        tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, gids=gids, n_wires=nw,
                       band_lengths=bl, norm_sigma=ns, cfg=cfg, value_clean=cl.value)    # tgt = clean
        B = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in to_fm(tok).items()}
        B["n_cells"] = B["plane_id"].shape[0]
        return B, bl

    def gather(keys, B, bl, feats):
        g, fw, ft = R.unkey(keys)
        w, t = fw * R.FW, ft * R.FT + R.FT // 2
        pc = pixel_cells(g, w, t, bl, cfg)
        ck = B["cell_key"].cpu().numpy(); order = np.argsort(ck); cks = ck[order]
        pos = np.clip(np.searchsorted(cks, pc), 0, len(cks) - 1)
        idx = np.where(cks[pos] == pc, order[pos], -1)
        D = feats.shape[1]
        X = torch.zeros((len(keys), nb * D), dtype=torch.float16, device=dev)
        for b in range(nb):
            m = idx[:, b] >= 0
            if m.any():
                X[torch.from_numpy(np.nonzero(m)[0]).to(dev), b * D:(b + 1) * D] = \
                    feats[torch.from_numpy(idx[m, b]).to(dev)].half()
        dec = (1 << np.asarray(cfg.lev)).astype(np.float64)
        toff = np.asarray(cfg.toff)[g % 3]
        offs = []
        for b in range(nb):
            tau = (t + toff) / dec[b] - cfg.delta[b]
            offs += [(w % pw + 0.5) / pw, (tau / pt) % 1.0]
        aux = np.concatenate([(idx >= 0).astype(np.float32), fourier(np.stack(offs, 1).astype(np.float32))], 1)
        return X.cpu(), aux.astype(np.float32), (idx >= 0).any(1)

    def region_mask(B):
        wc = (B["cell_wb"].float() * pw + pw / 2).cpu().numpy()
        tc = B["t_phys"].float().cpu().numpy()
        g = B["plane_id"].cpu().numpy()
        rk = (g.astype(np.int64) << 40) | ((wc // 32).astype(np.int64) << 20) | (np.floor(tc / 128).astype(np.int64) + 4096)
        h = np.array([zlib.crc32(int(x).to_bytes(8, "little", signed=True)) % 1000 for x in rk])
        return torch.from_numpy(h < 750).to(dev)

    @torch.no_grad()
    def recon(model, B, m):
        feat, rows = model.forward_feat(B, m, masked_only=True)
        NS, K = model.n_slot, model.n_bins
        acc = np.zeros(5)
        for s in range(0, rows.numel(), 2048):
            r, f = rows[s:s + 2048], feat[s:s + 2048]
            lp = torch.log_softmax(model.val_head(f).float().view(-1, NS, K) * model.readout_mult, -1)
            occ, valid, tgt, band = B["occ"][r].bool(), B["valid"][r].bool(), B["tgt"][r].float(), B["band_id"][r].long()
            sel = occ & valid
            rec = (lp.exp() * model.bin_cent_asinh[band][:, None, :].float()).sum(-1)
            y = tgt[sel]; d_ = rec[sel] - y
            bins = bucketize_bins(tgt, band, model.bin_edges, K)
            acc += [float((d_ * d_).sum()), float(y.sum()), float((y * y).sum()), float(sel.sum()),
                    float(-lp.gather(-1, bins[..., None]).squeeze(-1)[sel].sum())]
        return acc

    # ------------------------------------------------------------ extraction
    files = sorted(glob.glob(os.path.join(a.truth, "ev*.npz")))
    arms = ("trained", "random", "raw")
    data = {x: {k: [] for k in ("Xtr", "Atr", "ytr", "etr", "Xte", "Ate", "yte", "ete", "pte", "Xw", "Aw", "Ow")} for x in arms}
    win_meta, win_q, win_id, rec_acc = [], [], [], np.zeros(5)
    qs = np.concatenate([np.load(f)["mq"] for f in files[:40]])
    Q0 = float(np.median(qs[qs > 0]))
    t0 = time.time()
    for ei, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        test = int(os.path.basename(f)[2:5]) >= a.train_events
        B, bl = tokens(str(z["shard"]), int(z["event"]))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = {x: models[x].encode(B).float() for x in ("trained", "random")}
        feats["raw"] = torch.cat([B["inp"].float(), B["occ"].float()], 1)
        y = np.log1p(z["mq"] / Q0).astype(np.float32)
        for x in arms:
            X, A, ok = gather(z["mrows"], B, bl, feats[x])
            d = data[x]; s = "te" if test else "tr"
            d["X" + s].append(X[ok]); d["A" + s].append(A[ok]); d["y" + s].append(y[ok])
            d["e" + s].append(np.full(ok.sum(), ei))
            if test:
                d["pte"].append((R.unkey(z["mrows"][ok])[0] % 3).astype(np.int8))
                if len(z["wkey"]):
                    Xw, Aw, okw = gather(z["wkey"], B, bl, feats[x])
                    d["Xw"].append(Xw); d["Aw"].append(Aw); d["Ow"].append(okw)
        if test:
            if len(z["wkey"]):
                base = len(win_meta)
                mm = json.loads(str(z["meta"]))
                for m_ in mm:
                    m_["ev"] = ei
                win_meta += mm; win_q.append(z["wq"]); win_id.append(z["wid"] + base)
            rec_acc += recon(models["trained"], B, region_mask(B))
        if ei % 40 == 0:
            print(f"  event {ei}/{len(files)}  {time.time() - t0:.0f}s", flush=True)

    win_q, win_id = np.concatenate(win_q), np.concatenate(win_id)
    starts = np.searchsorted(win_id, np.arange(len(win_meta)))
    ends = np.append(starts[1:], len(win_id))
    sse, sy, syy, nv, ce_sum = rec_acc
    res = dict(tag=a.tag, checkpoint=a.checkpoint, pw=pw, pt=pt, d=models["trained"].d, Q0=Q0,
               train_events=a.train_events, n_windows=len(win_meta),
               recon_var_expl=float(1 - (sse / nv) / (syy / nv - (sy / nv) ** 2)), recon_val_ce=float(ce_sum / nv))

    # ------------------------------------------------------------ probes
    def fit(Xtr, Atr, ytr, etr):
        torch.manual_seed(0)                                   # the probe's init and batches, reproducible
        net = nn.Sequential(nn.Linear(Xtr.shape[1] + Atr.shape[1], 1024), nn.GELU(),
                            nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 1)).to(dev)
        va = etr >= np.quantile(etr, 0.9)                      # last ~10% of train events: early stop
        Xg, Ag, yg = Xtr.to(dev), torch.from_numpy(Atr).to(dev), torch.from_numpy(ytr).to(dev)
        tri, vai = torch.from_numpy(np.nonzero(~va)[0]).to(dev), torch.from_numpy(np.nonzero(va)[0]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        steps = 6 * len(tri) // 4096
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        best, best_sd = 1e9, None
        for s in range(steps):
            b = tri[torch.randint(len(tri), (4096,), device=dev)]
            loss = F.mse_loss(net(torch.cat([Xg[b].float(), Ag[b]], 1)).squeeze(-1), yg[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if (s + 1) % (steps // 6) == 0:
                with torch.no_grad():
                    v = sum(float(F.mse_loss(net(torch.cat([Xg[c].float(), Ag[c]], 1)).squeeze(-1), yg[c], reduction="sum"))
                            for c in vai.split(16384)) / len(vai)
                if v < best:
                    best, best_sd = v, {k: t_.clone() for k, t_ in net.state_dict().items()}
        net.load_state_dict(best_sd)
        return net, best

    @torch.no_grad()
    def predict(net, X, A):
        return torch.cat([net(torch.cat([X[s:s + 16384].to(dev).float(), torch.from_numpy(A[s:s + 16384]).to(dev)], 1))
                          .squeeze(-1).cpu() for s in range(0, X.shape[0], 16384)]).numpy()

    toff = {i: cfg.toff[i] for i in range(3)}
    for x in arms:
        d = data[x]
        cat = lambda k: torch.cat(d[k]) if isinstance(d[k][0], torch.Tensor) else np.concatenate(d[k])
        net, vloss = fit(cat("Xtr"), cat("Atr"), cat("ytr"), cat("etr"))
        p, y, e, pl = predict(net, cat("Xte"), cat("Ate")), cat("yte"), cat("ete"), cat("pte")
        q = np.expm1(y) * Q0; q30 = np.quantile(q[q > 0], 0.3)
        qh = np.expm1(np.maximum(predict(net, cat("Xw"), cat("Aw")), 0)) * Q0
        # A cell no token covers carries no model information: predict nothing there.
        # The probe is trained only on covered cells, so its output on an all-zero
        # input is an extrapolated constant -- which used to set every window's max
        # (noD2: the background 95th and 99th percentiles were the same number).
        qh[~cat("Ow")] = 0.0
        W = R.window_stats(qh, win_q, win_meta, starts, ends, pw, pt, cfg.delta[0], toff)
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
        with open(a.out.replace(".jsonl", f"_{a.tag}_{x}_windows.json"), "w") as fh:     # per-window scores
            json.dump([{k: w[k] for k in ("kind", "score", "eb", "dip", "edge", "sep") if k in w} for w in W], fh, default=float)
        rng = np.random.default_rng(0); boots = []
        for _ in range(a.boot):
            sm = rng.choice(evs, len(evs), replace=True)
            idx = np.concatenate([rows_of[v] for v in sm])
            boots.append(R.scalars(p[idx], q[idx], [z_ for v in sm for z_ in z_of[v]],
                                   [w for v in sm for w in wins_of[v]], q30))
        r = {"val_mse": vloss}
        for k, v in point.items():
            bv = np.array([b_[k] for b_ in boots], float)
            lo, hi = np.nanpercentile(bv, [16, 84]) if np.isfinite(bv).any() else (np.nan, np.nan)
            r[k] = [round(v, 4), round(float(lo), 4), round(float(hi), 4)]
        res[x] = r
        print(json.dumps({x: {k: r[k] for k in ("map_r", "faint_auc")}}), flush=True)

    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
