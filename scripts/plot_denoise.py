#!/usr/bin/env python3
"""Figures for a trained denoiser: input, truth, prediction, residual.

    python scripts/plot_denoise.py --checkpoint <run>/best.pt --truth <truth_v2> --out <dir> [--event 300] [--plane 2]

On the floor evaluation's grid (2 wires x 16 ticks) for one held-out probe event:
  full plane   noisy input | true pre-response charge | predicted | log10(pred/true) |
               false positives (prediction where the truth is empty)
  zoom         the same around a faint isolated deposit (0.1-0.2 MeV) and around a
               charge-free noise window, both taken from the evaluation's windows
  scatter      predicted vs true log1p(q/Q0) over every covered cell of the test events

Colour: truth and prediction share ONE logarithmic norm (charge spans ~3 decades),
floored at 0.05 Q0 with cells below drawn in the 'under' colour so faint charge is
not confused with empty; the input has its own log norm (|coefficient| in noise
sigma, a different quantity); the residual is a diverging log-ratio centred at 0,
shown only where the truth is above the floor -- a ratio against empty truth is
unbounded, so prediction on empty cells gets its own panel on the charge scale.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--truth", required=True, help="truth_v2 (probe split, eval windows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--event", type=int, default=None, help="truth file index (default: first test event with a faint iso window)")
    ap.add_argument("--plane", type=int, default=None, help="plane gid (default: the faint window's plane)")
    ap.add_argument("--scatter-events", type=int, default=48)
    a = ap.parse_args()

    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    import torch
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.model.tokenize import tick_of_tau
    from helix.probe import resolution as R

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta.get("head_kw") or {})
    model.load_state_dict(ck["model"]); model.to(dev).eval()
    presence = bool((meta.get("head_kw") or {}).get("presence"))
    Q0 = float(meta["q0"]); ds = DenoiseEvents("", [], "", items=[]); cfg = ds.cfg
    tag = os.path.basename(os.path.dirname(a.checkpoint))

    files = sorted(glob.glob(os.path.join(a.truth, "ev*.npz")))
    test = [f for f in files if int(os.path.basename(f)[2:5]) >= 260]

    def load(f):
        z = np.load(f, allow_pickle=True)
        shard, event = str(z["shard"]), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        ce, B, bl = ds.tokens(shard, pos)
        return z, ce, B, bl, shard, pos

    def predict(B, bl, keys):
        idx, aux = R.cell_inputs(keys, B["cell_key"], bl, cfg)
        ok = (idx >= 0).any(1); out = np.zeros(len(keys), np.float32)
        Bt = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.fm.encode(Bt)
            for s in range(0, int(ok.sum()), 65536):
                sel = np.nonzero(ok)[0][s:s + 65536]
                p = model.head(feats, torch.as_tensor(idx[sel]).to(dev), torch.as_tensor(aux[sel]).to(dev))
                out[sel] = (p[0] if presence else p).float().cpu().numpy()
        return out, ok

    def covered_cells(ce, g=None):
        """Every fine cell inside the footprint of a token (on plane g, or on every
        plane), vectorised over token blocks: each block of band b spans pw wires and
        pt << lev[b] ticks from its first tick, clamped at 0."""
        k = ce.band < cfg.n_bands if g is None else (ce.band < cfg.n_bands) & (ce.plane_gid == g)
        out = []
        for b in range(cfg.n_bands):
            m = k & (ce.band == b)
            if not m.any():
                continue
            gb, wb, tb = np.unique(np.stack([ce.plane_gid[m], ce.wire[m] // cfg.pw, ce.tau[m] // cfg.pt], 1), axis=0).T
            t_raw = tick_of_tau(tb * cfg.pt, gb, np.full(len(gb), b), cfg)
            f_lo = np.floor(np.maximum(t_raw, 0) / R.FT).astype(np.int64)
            f_hi = np.floor(np.maximum(t_raw + cfg.pt * (1 << cfg.lev[b]), 0) / R.FT).astype(np.int64)
            fw = (wb * cfg.pw // R.FW)[:, None] + np.arange(cfg.pw // R.FW)              # (n, wires)
            ft = f_lo[:, None] + np.arange(int((f_hi - f_lo).max()) + 1)                # (n, ticks)
            shape = (len(gb), fw.shape[1], ft.shape[1])
            sel = np.broadcast_to((ft <= f_hi[:, None])[:, None, :], shape)
            out.append(R.fkey(np.broadcast_to(gb[:, None, None], shape)[sel],
                              np.broadcast_to(fw[:, :, None], shape)[sel],
                              np.broadcast_to(ft[:, None, :], shape)[sel]))
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)

    # ---- choose the event and plane: a test event holding a faint isolated deposit
    choice = None
    for f in test if a.event is None else [files[a.event]]:
        mm = json.loads(str(np.load(f, allow_pickle=True)["meta"]))
        iso = [m for m in mm if m["kind"] == "iso" and 0.1 <= float(np.sum(m["E"])) < 0.2]
        bg = [m for m in mm if m["kind"] == "bg"]
        if iso and bg and (a.plane is None or any(m["g"] == a.plane for m in iso)):
            mi = next(m for m in iso if a.plane is None or m["g"] == a.plane)
            mb = next((m for m in bg if m["g"] == mi["g"]), bg[0])
            choice = (f, mi, mb); break
    f, mi, mb = choice
    g = int(mi["g"]) if a.plane is None else a.plane
    z, ce, B, bl, shard, pos = load(f)
    print(f"[plot] {tag}: {os.path.basename(f)} ({os.path.basename(shard)}#{pos}) plane {g}; faint window E={np.sum(mi['E']):.3f} MeV")

    # truth on the full grid for this plane (from the denoise truth store via the same decoding)
    from scripts.build_denoise_truth import event_cells  # noqa: E402
    SRC = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"
    hits = os.path.join(SRC, "hits", os.path.basename(os.path.dirname(shard)), ce.source_file.replace("_sensor_", "_hits_"))
    tk, tq = event_cells(hits, ce.event)
    tg, tw, tt = R.unkey(tk); on = tg == g
    cells = covered_cells(ce, g)
    pred_y, ok = predict(B, bl, cells)
    cg, cw, ct = R.unkey(cells)
    nw, nt = int(max(cw.max(), tw[on].max() if on.any() else 0)) + 1, int(max(ct.max(), tt[on].max() if on.any() else 0)) + 1
    T = np.zeros((nw, nt)); T[tw[on], tt[on]] = tq[on]
    P = np.zeros((nw, nt)); P[cw[ok], ct[ok]] = Q0 * np.expm1(np.maximum(pred_y[ok], 0))
    k = (ce.band < cfg.n_bands) & (ce.plane_gid == g)
    ctick = tick_of_tau(ce.tau[k], ce.plane_gid[k], ce.band[k], cfg)
    from helix.model.tokenize import sigma_for_rows
    meta_c = ds._shard_cfg(shard)
    sig = sigma_for_rows(ce.plane_gid[k], ce.band[k], meta_c["gids"], meta_c["norm_sigma"])
    I = np.zeros((nw, nt))
    np.add.at(I, (np.clip(ce.wire[k] // R.FW, 0, nw - 1), np.clip((ctick // R.FT).astype(int), 0, nt - 1)), np.abs(ce.value[k]) / sig)

    floor = 0.05 * Q0
    vmax = np.quantile(np.concatenate([T[T > 0], P[P > 0]]), 0.999)
    qnorm = LogNorm(vmin=floor, vmax=vmax)
    cmap = plt.get_cmap("viridis").copy(); cmap.set_under("#1b1b2f"); cmap.set_bad("#1b1b2f")
    icmap = plt.get_cmap("magma").copy(); icmap.set_under("black")
    inorm = LogNorm(vmin=3, vmax=max(np.quantile(I[I > 0], 0.999), 4))

    def panels(sl_w, sl_t, title, fn, boxes=()):
        fig, ax = plt.subplots(1, 5, figsize=(27, 6), sharex=True, sharey=True)
        ext = [sl_t.start * R.FT, sl_t.stop * R.FT, sl_w.stop * R.FW, sl_w.start * R.FW]
        Ti, Pi, Ii = T[sl_w, sl_t], P[sl_w, sl_t], I[sl_w, sl_t]
        im0 = ax[0].imshow(np.where(Ii > 0, Ii, np.nan), aspect="auto", extent=ext, cmap=icmap, norm=inorm, interpolation="nearest")
        ax[0].set_title("noisy input: sum |coefficient| per cell (noise sigma)")
        im1 = ax[1].imshow(np.where(Ti > 0, Ti, np.nan), aspect="auto", extent=ext, cmap=cmap, norm=qnorm, interpolation="nearest")
        ax[1].set_title("TRUE pre-response charge (hits)")
        ax[2].imshow(np.where(Pi > 0, Pi, np.nan), aspect="auto", extent=ext, cmap=cmap, norm=qnorm, interpolation="nearest")
        ax[2].set_title("PREDICTED charge (same scale)")
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(Ti > floor, np.log10(np.maximum(Pi, floor / 10) / Ti), np.nan)
        im3 = ax[3].imshow(lr, aspect="auto", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1.5, 1.5), interpolation="nearest")
        ax[3].set_title("log10(pred / true), true > 0.05 Q0")
        fp = np.where((Ti == 0) & (Pi > 0), Pi, np.nan)
        ax[4].imshow(fp, aspect="auto", extent=ext, cmap=cmap, norm=qnorm, interpolation="nearest")
        ax[4].set_title(f"false positives (true = 0): {int(np.sum(fp > floor)):,} cells > 0.05 Q0")
        for x in ax:
            x.set_xlabel("tick"); x.set_facecolor("#1b1b2f")
            for (bw, bt, c) in boxes:
                x.add_patch(plt.Rectangle((bt, bw), R.WT, R.WW, fill=False, ec=c, lw=1.5))
        ax[0].set_facecolor("black"); ax[3].set_facecolor("white"); ax[0].set_ylabel("wire")
        fig.colorbar(im0, ax=ax[0], fraction=0.04); fig.colorbar(im1, ax=ax[1:3], fraction=0.02, label="charge (e-)")
        fig.colorbar(im3, ax=ax[3], fraction=0.04, label="log10 ratio")
        fig.suptitle(title); fig.savefig(os.path.join(a.out, fn), dpi=110, bbox_inches="tight"); plt.close(fig)
        print("[plot] wrote", fn)

    name = f"{tag}_{os.path.basename(f)[:-4]}_g{g}"
    panels(slice(0, nw), slice(0, nt), f"{tag} -- {os.path.basename(shard)}#{pos}, plane {g}: full plane "
           f"(cyan box: faint isolated deposit {np.sum(mi['E']):.2f} MeV; orange: a noise window)", f"{name}_full.png",
           boxes=[(mi["w0"], mi["t0"], "cyan"), (mb["w0"], mb["t0"], "orange")] if mb["g"] == g else [(mi["w0"], mi["t0"], "cyan")])
    for m, lab in ((mi, "faint"), (mb, "noise")):
        if m["g"] != g:
            continue
        w0, t0 = m["w0"] // R.FW, m["t0"] // R.FT; pad_w, pad_t = 24, 24
        sw = slice(max(w0 - pad_w, 0), min(w0 + R.WW // R.FW + pad_w, nw)); st = slice(max(t0 - pad_t, 0), min(t0 + R.WT // R.FT + pad_t, nt))
        win = P[w0:w0 + R.WW // R.FW, t0:t0 + R.WT // R.FT]
        panels(sw, st, f"{tag} -- zoom on the {lab} window ({'E=%.3f MeV' % np.sum(m['E']) if lab == 'faint' else 'charge-free'}); "
               f"max predicted in window {win.max():.0f} e-", f"{name}_zoom_{lab}.png", boxes=[(m["w0"], m["t0"], "cyan" if lab == "faint" else "orange")])

    # ---- predicted vs true over every covered cell of the test events
    YT, YP = [], []
    for f2 in test[:a.scatter_events]:
        z2, ce2, B2, bl2, sh2, pos2 = load(f2)
        h2 = os.path.join(SRC, "hits", os.path.basename(os.path.dirname(sh2)), ce2.source_file.replace("_sensor_", "_hits_"))
        tk2, tq2 = event_cells(h2, ce2.event)
        cells2 = covered_cells(ce2)
        yp, ok2 = predict(B2, bl2, cells2)
        pos_ = np.clip(np.searchsorted(tk2, cells2), 0, max(len(tk2) - 1, 0))
        q = np.where(tk2[pos_] == cells2, tq2[pos_], 0.0) if len(tk2) else np.zeros(len(cells2))
        YT.append(np.log1p(q[ok2] / Q0)); YP.append(yp[ok2])
    yt, yp = np.concatenate(YT), np.concatenate(YP)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    hmax = max(yt.max(), yp.max())
    h = ax[0].hist2d(yt, yp, bins=200, range=[[0, hmax], [min(yp.min(), 0), hmax]], norm=LogNorm(), cmap="viridis")
    ax[0].plot([0, hmax], [0, hmax], "w--", lw=1)
    ax[0].set_xlabel("true log1p(q / Q0)"); ax[0].set_ylabel("predicted log1p(q / Q0)")
    ax[0].set_title(f"every covered cell, {len(YT)} test events ({len(yt):,} cells)\nempty cells sit at x = 0")
    fig.colorbar(h[3], ax=ax[0], label="cells")
    e0 = yp[yt == 0]
    ax[1].hist(e0, bins=200, range=(min(e0.min(), -0.2), max(e0.max(), 1.0)), log=True, color="tab:orange", alpha=0.8, label=f"empty cells ({len(e0):,})")
    ax[1].hist(yp[(yt > 0) & (yt < np.log1p(0.5))], bins=200, range=(min(e0.min(), -0.2), max(e0.max(), 1.0)), log=True, color="tab:blue", alpha=0.6, label="faint cells (q < 0.5 Q0)")
    ax[1].set_xlabel("predicted log1p(q / Q0)"); ax[1].set_ylabel("cells"); ax[1].legend()
    ax[1].set_title("prediction on empty vs faint cells -- the overlap is the floor")
    fig.suptitle(f"{tag}: predicted vs true"); fig.savefig(os.path.join(a.out, f"{tag}_scatter.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    print("[plot] wrote", f"{tag}_scatter.png")


if __name__ == "__main__":
    main()
