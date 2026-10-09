#!/usr/bin/env python3
"""Side-by-side figures of several denoisers on the same regions.

    python scripts/plot_compare.py --out <dir> --model NAME=CKPT[:raw|:gated] ... [--region ev,g,w0,w1,t0,t1,label ...]

Columns: noisy input, truth, then one column per model entry -- ``:gated`` keeps
the charge map only where the model's presence head says > 0.5, ``:raw`` shows the
charge head as is. Rows: charge (one LogNorm in electrons shared by truth and every
model, floored at 0.05 Q0), log10(pred / true) on truth-charged cells, and the
prediction on truth-empty cells (false charge) with its count above 0.05 Q0. Regions
are in raw wires / ticks of plane gid ``g`` of truth file ``ev``; cells no token
covers are 0, as in the evaluation.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_REGIONS = [
    "260,1,0,3600,0,4400,full plane (ev260 V)",
    "260,1,1640,1760,400,1300,faint 0.20 MeV deposit (ev260 V)",
    "300,2,610,740,320,1220,0.15 MeV deposit beside a track (ev300 Y)",
    "311,3,584,664,3680,4320,dense shower (ev311 U)",
    "296,4,560,640,930,1570,dense shower (ev296 V)",
    "293,5,778,858,3650,4290,dense shower (ev293 Y)",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    W = "/pscratch/sd/o/oalter/helix_work"
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT[:raw|:gated]")
    ap.add_argument("--region", action="append", default=None, help="ev,g,w0,w1,t0,t1,label (raw wires/ticks)")
    ap.add_argument("--truth", default=f"{W}/resolution/truth_v2")
    ap.add_argument("--source", default="/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02")
    a = ap.parse_args()

    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    import torch
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.model.tokenize import sigma_for_rows, tick_of_tau
    from helix.probe import resolution as R
    from scripts.build_denoise_truth import event_cells

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    ds = DenoiseEvents("", [], "", items=[]); cfg = ds.cfg
    models = []
    for spec in a.model:
        name, rest = spec.split("=", 1)
        ck_path, mode = (rest.rsplit(":", 1) if rest.endswith((":raw", ":gated")) else (rest, "gated"))
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
        hk = meta.get("head_kw") or {}
        m = build_denoise(meta["arch"], None, overrides=ov, head_kw=hk)
        m.load_state_dict(ck["model"]); m.to(dev).eval()
        models.append(dict(name=name, model=m, presence=bool(hk.get("presence")), decoder=hk.get("kind") == "decoder",
                           mode=mode, q0=float(meta["q0"])))
    Q0 = models[0]["q0"]

    def covered_cells(ce, g):
        k = (ce.band < cfg.n_bands) & (ce.plane_gid == g)
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
            out.append(R.fkey(np.broadcast_to(gb[:, None, None], shape)[sel], np.broadcast_to(fw[:, :, None], shape)[sel],
                              np.broadcast_to(ft[:, None, :], shape)[sel]))
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)

    def predict(md, B, bl, keys):
        idx, aux = R.cell_inputs(keys, B["cell_key"], bl, cfg)
        ok = (idx >= 0).any(1)
        q = np.zeros(len(keys), np.float32); pr = np.ones(len(keys), np.float32)
        Bt = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        sel_all = np.nonzero(ok)[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = md["model"].fm.encode(Bt)
            for s in range(0, len(sel_all), 65536):
                sel = sel_all[s:s + 65536]
                ti, ta = torch.as_tensor(idx[sel]).to(dev), torch.as_tensor(aux[sel]).to(dev)
                if md["decoder"]:
                    nb = torch.as_tensor(R.cell_neighbors(keys[sel], B["cell_key"], bl, cfg)).to(dev)
                    p = md["model"].head(feats, ti, ta, nbr=nb, B=Bt)
                else:
                    p = md["model"].head(feats, ti, ta)
                if md["presence"]:
                    pr[sel] = torch.sigmoid(p[1].float()).cpu().numpy(); p = p[0]
                q[sel] = Q0 * np.expm1(np.maximum(p.float().cpu().numpy(), 0))
        if md["presence"] and md["mode"] == "gated":
            q = np.where(pr > 0.5, q, 0.0)
        return q

    cache = {}

    def event_planes(ev, g):
        if (ev, g) in cache:
            return cache[(ev, g)]
        z = np.load(os.path.join(a.truth, f"ev{ev:03d}.npz"), allow_pickle=True)
        shard, event = str(z["shard"]), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        ce, B, bl = ds.tokens(shard, pos)
        run = os.path.basename(os.path.dirname(shard))
        tk, tq = event_cells(os.path.join(a.source, "hits", run, ce.source_file.replace("_sensor_", "_hits_")), ce.event)
        tg, tw, tt = R.unkey(tk); on = tg == g
        cells = covered_cells(ce, g); cg, cw, ct = R.unkey(cells)
        nw = int(max(cw.max(), tw[on].max() if on.any() else 0)) + 1
        nt = int(max(ct.max(), tt[on].max() if on.any() else 0)) + 1
        T = np.zeros((nw, nt)); T[tw[on], tt[on]] = tq[on]
        k = (ce.band < cfg.n_bands) & (ce.plane_gid == g)
        ctick = tick_of_tau(ce.tau[k], ce.plane_gid[k], ce.band[k], cfg)
        mc = ds._shard_cfg(shard)
        sig = sigma_for_rows(ce.plane_gid[k], ce.band[k], mc["gids"], mc["norm_sigma"])
        I = np.zeros((nw, nt))
        np.add.at(I, (np.clip(ce.wire[k] // R.FW, 0, nw - 1), np.clip((ctick // R.FT).astype(int), 0, nt - 1)),
                  np.abs(ce.value[k]) / sig)
        Ps = []
        for md in models:
            P = np.zeros((nw, nt)); P[cw, ct] = predict(md, B, bl, cells); Ps.append(P)
        cache[(ev, g)] = (T, I, Ps)
        return cache[(ev, g)]

    floor = 0.05 * Q0
    cmap = plt.get_cmap("viridis").copy(); cmap.set_under("#1b1b2f"); cmap.set_bad("#1b1b2f")
    icmap = plt.get_cmap("magma").copy(); icmap.set_under("black"); icmap.set_bad("black")
    for ri, spec in enumerate(a.region or DEFAULT_REGIONS):
        ev, g, w0, w1, t0, t1, label = spec.split(",", 6)
        ev, g, w0, w1, t0, t1 = (int(x) for x in (ev, g, w0, w1, t0, t1))
        T, I, Ps = event_planes(ev, g)
        sw = slice(max(w0 // R.FW, 0), min(w1 // R.FW, T.shape[0])); st = slice(max(t0 // R.FT, 0), min(t1 // R.FT, T.shape[1]))
        ext = [st.start * R.FT, st.stop * R.FT, sw.stop * R.FW, sw.start * R.FW]
        Ti, Ii = T[sw, st], I[sw, st]
        vmax = max(np.quantile(np.concatenate([Ti[Ti > 0]] + [P[sw, st][P[sw, st] > 0] for P in Ps] + [np.array([floor * 10])]), 0.999), floor * 10)
        qn = LogNorm(vmin=floor, vmax=vmax)
        ncol = 2 + len(models)
        fig, ax = plt.subplots(3, ncol, figsize=(4.2 * ncol, 11), sharex=True, sharey=True)
        ax[0, 0].imshow(np.where(Ii > 0, Ii, np.nan), aspect="auto", extent=ext, cmap=icmap,
                        norm=LogNorm(vmin=3, vmax=max(np.quantile(Ii[Ii > 0], 0.999) if (Ii > 0).any() else 4, 4)), interpolation="nearest")
        ax[0, 0].set_title("noisy input\n(sum |coefficient| / noise sigma)")
        im = ax[0, 1].imshow(np.where(Ti > 0, Ti, np.nan), aspect="auto", extent=ext, cmap=cmap, norm=qn, interpolation="nearest")
        ax[0, 1].set_title(f"TRUE charge\nsum {Ti.sum() / 1e6:.2f}M e-")
        for j, (md, P) in enumerate(zip(models, Ps)):
            Pi = P[sw, st]; c = 2 + j
            ax[0, c].imshow(np.where(Pi > 0, Pi, np.nan), aspect="auto", extent=ext, cmap=cmap, norm=qn, interpolation="nearest")
            ax[0, c].set_title(f"{md['name']}\nsum {Pi.sum() / 1e6:.2f}M e-")
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.where(Ti > floor, np.log10(np.maximum(Pi, floor / 10) / Ti), np.nan)
            imr = ax[1, c].imshow(lr, aspect="auto", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1.5, 1.5), interpolation="nearest")
            med = np.nanmedian(np.abs(lr)) if np.isfinite(lr).any() else np.nan
            ax[1, c].set_title(f"log10(pred/true), true > 0.05 Q0\nmedian |log10| {med:.2f}")
            fp = np.where((Ti == 0) & (Pi > 0), Pi, np.nan)
            ax[2, c].imshow(fp, aspect="auto", extent=ext, cmap=cmap, norm=qn, interpolation="nearest")
            ax[2, c].set_title(f"false charge (true = 0)\n{int(np.nansum(fp > floor)):,} cells > 0.05 Q0, "
                               f"{int(np.nansum(fp > Q0)):,} > 1 Q0")
        for r in (1, 2):
            for c in (0, 1):
                ax[r, c].axis("off")
        for x in ax.ravel():
            x.set_facecolor("#1b1b2f")
        for x in ax[1, 2:]:
            x.set_facecolor("white")
        for x in ax[-1]:
            x.set_xlabel("tick")
        for x in ax[:, 0]:
            x.set_ylabel("wire")
        fig.colorbar(im, ax=ax[0, :], fraction=0.015, label="charge per cell (e-), 2 wires x 16 ticks")
        fig.colorbar(imr, ax=ax[1, :], fraction=0.015, label="log10 ratio")
        fig.suptitle(f"{label}: ev{ev:03d}, plane gid {g}, wires {w0}-{w1}, ticks {t0}-{t1}", fontsize=13)
        fn = os.path.join(a.out, f"compare_{ri}_ev{ev:03d}_g{g}.png")
        fig.savefig(fn, dpi=95, bbox_inches="tight"); plt.close(fig)
        print("[compare] wrote", fn, flush=True)


if __name__ == "__main__":
    main()
