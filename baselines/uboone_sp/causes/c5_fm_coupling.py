#!/usr/bin/env python3
"""C5: is the amplitude coupling already in the PRETRAINED foundation model, in its own masked-prediction task?

One fixed random mask hides 25% of the tokens of each volume. Visible coefficients are scaled by 1.05 in a region;
the change of the FM's predicted value (mu, arcsinh units ~ log amplitude for bright slots) on HIDDEN tokens, per
bright occupied slot (|target| > 4, i.e. |v / sigma| > 27), signed by the target, / log(1.05):
  vol0     scale all visible volume-0 coefficients: response of hidden volume-1 tokens ('cross-volume'); the data
           correlate the volumes' charge at +0.17, so an ideal predictor responds >= 0
  planeU0  scale visible U coefficients of volume 0 (gid 0): response of hidden V / Y tokens of volume 0
           ('cross-plane'); the planes see the same charge (totals correlate 1.0), so an ideal predictor responds > 0
  and the response of hidden tokens INSIDE the scaled region (gid 0 / volume 0) for reference.
Also the same response for the hidden tokens' slots of all amplitudes.

    CUDA_VISIBLE_DEVICES=3 scripts/helix_run.sh python baselines/uboone_sp/causes/c5_fm_coupling.py --out <json>
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
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
ART = "/pscratch/sd/o/oalter/helix_work/probe/artifact_nz_pw16_s1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-271")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.model.artifact import load
    from helix.model.fm import build_fm
    from helix.model.tokenize import PatchConfig, assemble, to_fm
    dev = torch.device("cuda")
    art = load(ART)
    arch = dict(art.arch); arch["compile_blocks"] = False
    fm = build_fm(arch)
    fm.load_state_dict(art.state_dict); fm.to(dev).eval()
    print("FM", type(fm).__name__, "n_bins", getattr(fm, "n_bins", None), "dec_mode", getattr(fm, "dec_mode", None),
          "centroids finite", bool(torch.isfinite(fm.bin_cent_asinh).all()) if getattr(fm, "n_bins", 0) else None, flush=True)
    cfg = PatchConfig(cell_t="grid_center")
    acc = {k: [] for k in ("vol0_cross", "vol0_inside", "planeU0_cross", "planeU0_inside",
                           "vol0_cross_all", "planeU0_cross_all")}
    lo, hi = map(int, a.events.split("-"))
    for ev in range(lo, hi + 1):
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        run = os.path.basename(os.path.dirname(str(z["shard"])))
        path = os.path.join(CORPUS, run, os.path.basename(str(z["shard"])))
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
            c = {k: fh["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        ce = read_coeff_event(path, pos)

        def tokens(scale_mask):
            v = ce.value.astype(np.float64) * (1.0 if scale_mask is None else np.where(scale_mask, 1.05, 1.0))
            return to_fm(assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, v.astype(np.float32), gids=c["gids"],
                                  n_wires=c["n_wires"], band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg))

        B0 = tokens(None)
        gid = np.asarray(B0["plane_id"]); vol = gid // 3
        rng = np.random.default_rng(ev)
        hide = rng.random(len(gid)) < 0.25                                   # same mask for every variant
        tgt = np.asarray(B0["inp"]); occ = np.asarray(B0["occ"]) > 0

        @torch.no_grad()
        def mu_of(B):
            Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
            Bt["n_cells"] = Bt["plane_id"].shape[0]
            o = fm.raw_heads(Bt, torch.as_tensor(hide).to(dev))
            m_ = o[1].float()
            if m_.dim() == 3 and m_.shape[-1] > 1:                 # categorical head: expected value over bin centroids
                cent = fm.bin_cent_asinh.float().to(m_.device)          # (n_band, K) or (K,)
                cent = cent[Bt["band_id"].long()][:, None, :] if cent.dim() == 2 else cent
                m_ = (m_.softmax(-1) * cent).sum(-1)
            return m_.cpu().numpy()                                    # (n_tok, n_slot) predicted arcsinh value
        mu0 = mu_of(B0)
        for vname, smask, inside_tok, cross_tok in (
                ("vol0", (ce.plane_gid // 3) == 0, vol == 0, vol == 1),
                ("planeU0", ce.plane_gid == 0, gid == 0, (gid == 1) | (gid == 2))):
            mu1 = mu_of(tokens(smask))
            d = (mu1 - mu0) * np.sign(tgt) / np.log(1.05)
            bright = occ & (np.abs(tgt) > 4)
            for nm, tok in (("inside", inside_tok), ("cross", cross_tok)):
                sel = bright & (hide & tok)[:, None]
                acc[f"{vname}_{nm}"].append(d[sel])
            acc[f"{vname}_cross_all"].append(d[occ & (hide & cross_tok)[:, None]])
        print(f"ev{ev:03d} {len(gid)} tokens", flush=True)
    out = {}
    for k, v in acc.items():
        x = np.concatenate(v)
        a16, a50, a84 = np.percentile(x, [16, 50, 84])
        out[k] = dict(med=float(a50), mean=float(x.mean()), hw68=float((a84 - a16) / 2), n=int(len(x)))
        print(k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in out[k].items()}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
