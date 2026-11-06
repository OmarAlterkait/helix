#!/usr/bin/env python3
"""C9: two facts the representation questions rest on.

(a) Cathode: the two volumes share the cathode at x = 0 (JAXTPC cubic_wireplane_config: x < 0 drifts to -x, x > 0
    to +x), so equal drift time means mirror positions +-x, adjacent only near the cathode (tick ~3510). Does a wire
    index mean the same y-z line in both volumes? Per event, complete-truth charge on the two Y planes (gid 2, 5)
    within the cathode window, as wire profiles; correlation same-index vs mirrored (n_wires - 1 - w).
(b) Per layer of the PRETRAINED FM: how precisely can a linear readout recover each token's OWN bright coefficient
    values (arcsinh units = log amplitude) from that layer's features? Ridge per layer fitted on calibration events
    (ev000-031), scored on test events (ev260-275); hw68 of the residual on bright occupied slots (|value| > 4), vs
    the categorical bins' width (A4 0.082, D4 0.071, D3 0.063, D2 0.050 in log amplitude; uniform rms = width/sqrt 12).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
ART = "/pscratch/sd/o/oalter/helix_work/probe/artifact_nz_pw16_s1"


def item(ev):
    import h5py
    z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
    run = os.path.basename(os.path.dirname(str(z["shard"])))
    path = os.path.join(CORPUS, run, os.path.basename(str(z["shard"])))
    with h5py.File(path, "r") as fh:
        pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
        c = {k: fh["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
    return run, path, pos, c


def cathode(evs):
    from helix.data.denoise import DenoiseEvents
    from helix.probe.resolution import FT, FW, unkey
    ds = DenoiseEvents(CORPUS, [], CT, items=[])
    same, mirr, nz = [], [], 0
    for ev in evs:
        run, path, pos, c = item(ev)
        k, q, _ = ds._truth(run, path, pos)
        g, fw, ft = unkey(k)
        tick = ft * FT
        win = (tick >= 3350) & (tick < 3650)
        nw = 1443 // FW + 1
        p2 = np.bincount(fw[win & (g == 2)], weights=q[win & (g == 2)], minlength=nw)[:nw]
        p5 = np.bincount(fw[win & (g == 5)], weights=q[win & (g == 5)], minlength=nw)[:nw]
        if p2.sum() < 1e5 or p5.sum() < 1e5:
            continue
        nz += 1
        same.append(np.corrcoef(p2, p5)[0, 1]); mirr.append(np.corrcoef(p2, p5[::-1])[0, 1])
    return dict(n_events_with_cathode_charge_both=nz, corr_same_index=float(np.nanmean(same)) if same else None,
                corr_mirrored=float(np.nanmean(mirr)) if mirr else None)


def layers(cal, test):
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.model.artifact import load
    from helix.model.fm import build_fm
    from helix.model.tokenize import PatchConfig, assemble, to_fm
    dev = torch.device("cuda")
    art = load(ART); arch = dict(art.arch); arch["compile_blocks"] = False
    fm = build_fm(arch); fm.load_state_dict(art.state_dict); fm.to(dev).eval()
    cfg = PatchConfig(cell_t="grid_center")
    L = len(fm.enc)

    @torch.no_grad()
    def feats(ev):
        run, path, pos, c = item(ev)
        ce = read_coeff_event(path, pos)
        B = to_fm(assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, gids=c["gids"], n_wires=c["n_wires"],
                           band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg))
        Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        X = fm.encode_layers(Bt, list(range(1, L + 1)))
        X[0] = fm._emb(Bt)
        inp, occ = np.asarray(B["inp"]), np.asarray(B["occ"]) > 0
        keep = (np.abs(inp) * occ).max(1) > 4                                   # tokens with a bright slot
        return {k: v[torch.as_tensor(keep).to(dev)].double() for k, v in X.items()}, inp[keep], occ[keep], np.asarray(B["band_id"])[keep]

    XtX = {k: None for k in range(L + 1)}; XtY = {k: None for k in range(L + 1)}
    for ev in cal:
        X, inp, occ, band = feats(ev)
        Y = torch.as_tensor(inp, device=dev, dtype=torch.float64)
        for k, x in X.items():
            xa = torch.cat([x, torch.ones(len(x), 1, device=dev, dtype=x.dtype)], 1)
            XtX[k] = xa.T @ xa if XtX[k] is None else XtX[k] + xa.T @ xa
            XtY[k] = xa.T @ Y if XtY[k] is None else XtY[k] + xa.T @ Y
    W = {}
    for k in XtX:
        A = XtX[k]; lam = 1e-4 * torch.diagonal(A)[:-1].mean()
        A = A + torch.diag(torch.cat([torch.full((A.shape[0] - 1,), float(lam), device=dev), torch.zeros(1, device=dev)])).double()
        W[k] = torch.linalg.solve(A, XtY[k])
    res = {k: {b: [] for b in range(4)} for k in XtX}
    for ev in test:
        X, inp, occ, band = feats(ev)
        for k, x in X.items():
            xa = torch.cat([x, torch.ones(len(x), 1, device=dev, dtype=x.dtype)], 1)
            P = (xa @ W[k]).cpu().numpy()
            m = occ & (np.abs(inp) > 4)
            for b in range(4):
                mb = m & (band == b)[:, None]
                res[k][b].append((P - inp)[mb])
    out = {}
    for k in res:
        out[k] = {}
        for b in range(4):
            e = np.concatenate(res[k][b])
            a16, a84 = np.percentile(e, [16, 84])
            out[k][f"band{b}"] = dict(hw68=float((a84 - a16) / 2), n=int(len(e)))
        print("layer", k, {b: round(v["hw68"], 4) for b, v in out[k].items()}, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"cathode": cathode(list(range(0, 64)) + list(range(260, 308)))}
    print("cathode", res["cathode"], flush=True)
    res["layers"] = layers(list(range(0, 32)), list(range(260, 276)))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
