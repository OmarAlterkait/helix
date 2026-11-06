#!/usr/bin/env python3
"""Why does M7dc couple each cell to the event-wide amplitude (T2d)? Two direct checks.

C1b (simulation, no model): per event, median and spread of corpus-Tikhonov / complete truth on bright Y cells
     (floor.py maps; 48 test + 63 calibration events). A per-event amplitude-to-charge scale in the simulation
     would show as event-to-event spread of the median well beyond its statistical error.
C2   (models): scale TPC volume 0's coefficients by 1.05; bright cells (T >= 16 k e-, training-label complete truth)
     in volume 0 ('inside') and volume 1 ('control'), response = dy / dlog(1.05) in fp32. Models with different
     histories: pretrained-then-fine-tuned vs trained from scratch (matched pair le3_ft / le3_scratch, 16k labels;
     full-scale M4n_ft vs M2n / M3 scratch), old CellHead vs CellDecoder, fast (4k steps) vs full.

    CUDA_VISIBLE_DEVICES=3 scripts/helix_run.sh python baselines/uboone_sp/causes/c2_coupling.py --out <json>
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
W = "/pscratch/sd/o/oalter/helix_work/denoise"
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
Q0 = 2043.2578108909702
MODELS = {
    "M7dc (decoder+hurdle, ft, full)": f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt",
    "M6d (decoder, ft, full, hits label)": f"{R}/denoise/M6d_ft_full_dec/final.pt",
    "sp_dec_c25 (decoder, ft, 4k steps)": f"{R}/denoise/sp_dec_c25/best.pt",
    "M4n (CellHead, ft, full)": f"{W}/M4n_ft_full_near/final.pt",
    "M2n (CellHead, scratch, full)": f"{W}/M2n_scratch_full_near/final.pt",
    "M3 (CellHead+presence, scratch, full)": f"{W}/M3_scratch_full_win_pres/last.pt",
    "le3_ft_N16384 (CellHead, ft, 16k labels)": f"{W}/le3_ft_N16384/final.pt",
    "le3_scratch_N16384 (CellHead, scratch, 16k labels)": f"{W}/le3_scratch_N16384/final.pt",
}


def c1b():
    out = {}
    for name, evs, d in (("test", range(260, 308), f"{R}/bright/floor"), ("cal", [e for e in range(64) if e != 41], f"{R}/m7diag/cal_floor")):
        meds, errs = [], []
        for ev in evs:
            f = os.path.join(d, f"ev{ev:03d}.npz")
            if not os.path.exists(f):
                continue
            F = np.load(f)
            r = np.concatenate([F[f"corpus_tik0.001_g{g}"][F[f"truth_g{g}"] >= 64e3] / F[f"truth_g{g}"][F[f"truth_g{g}"] >= 64e3]
                                for g in (2, 5)])
            if len(r) < 50:
                continue
            a16, a50, a84 = np.percentile(r, [16, 50, 84])
            meds.append(a50); errs.append(1.2533 * (a84 - a16) / 2 / np.sqrt(len(r)))
        meds, errs = np.array(meds), np.array(errs)
        out[name] = dict(n_events=int(len(meds)), median_of_medians=float(np.median(meds)), sd_event_medians=float(meds.std()),
                         typical_stat_error=float(np.median(errs)),
                         excess_sd=float(np.sqrt(max(meds.var() - (errs ** 2).mean(), 0))))
        print("C1b", name, out[name], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-271")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"C1b": c1b(), "C2": {}}
    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import CellDecoder, build_denoise
    from helix.model.tokenize import assemble, to_fm
    from helix.probe import resolution as R_
    dev = torch.device("cuda")
    ds = DenoiseEvents(CORPUS, [], CT, items=[]); cfg = ds.cfg
    lo, hi = map(int, a.events.split("-"))
    ev_data = []
    for ev in range(lo, hi + 1):
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        run = os.path.basename(os.path.dirname(str(z["shard"])))
        path = os.path.join(CORPUS, run, os.path.basename(str(z["shard"])))
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
            c = {k: fh["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        tkeys, tq, _ = ds._truth(run, path, pos)
        sel = tq >= 16e3
        ev_data.append((read_coeff_event(path, pos), c, tkeys[sel].astype(np.int64), tq[sel]))
    def run_model(ckpt, t0):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
        model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta.get("head_kw") or {})
        model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
        decoder = isinstance(model.head, CellDecoder)
        S = {"inside": [], "control": [], "all": []}
        for ce, c, keys, T in ev_data:
            g, _, _ = R_.unkey(keys)
            ys = {}
            for vname, m in (("base", None), ("vol0", (ce.plane_gid // 3) == 0), ("all", np.ones(len(ce.value), bool))):
                v = ce.value.astype(np.float64) * (1.0 if m is None else np.where(m, 1.05, 1.0))
                B = to_fm(assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, v.astype(np.float32), gids=c["gids"], n_wires=c["n_wires"],
                                   band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg))
                idx, aux = R_.cell_inputs(keys, B["cell_key"], c["band_lengths"], cfg)
                ok = (idx >= 0).any(1)
                Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
                Bt["n_cells"] = Bt["plane_id"].shape[0]
                y = np.full(len(keys), np.nan)
                with torch.no_grad():
                    feats = model.fm.encode(Bt)
                    ti, ta = torch.as_tensor(idx[ok]).to(dev), torch.as_tensor(aux[ok]).to(dev)
                    if decoder:
                        nbr = torch.as_tensor(R_.cell_neighbors(keys[ok], B["cell_key"], c["band_lengths"], cfg)).to(dev)
                        o = model.head(feats, ti, ta, nbr=nbr, B=Bt)
                    else:
                        o = model.head(feats, ti, ta)
                    o = o[0] if isinstance(o, tuple) else o
                    y[ok] = o.float().cpu().numpy()
                ys[vname] = y
            q0 = Q0 * np.expm1(np.maximum(ys["base"], 0))
            lin = np.log1p(q0 * 1.05 / Q0) - np.log1p(q0 / Q0)
            S["inside"].append(((ys["vol0"] - ys["base"]) / lin)[g // 3 == 0])
            S["control"].append(((ys["vol0"] - ys["base"]) / lin)[g // 3 == 1])
            S["all"].append((ys["all"] - ys["base"]) / lin)
        r = {}
        for k, v in S.items():
            x = np.concatenate(v); x = x[np.isfinite(x)]
            a16, a50, a84 = np.percentile(x, [16, 50, 84])
            r[k] = dict(med=float(a50), hw68=float((a84 - a16) / 2), n=int(len(x)))
        print("C2", ckpt.split("/")[-2], {k: round(v["med"], 3) for k, v in r.items()}, f"{time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
        return r
    for name, ckpt in MODELS.items():
        if not os.path.exists(ckpt):
            print("missing", name, ckpt, flush=True); continue
        t0 = time.time()
        try:
            res["C2"][name] = run_model(ckpt, t0)
        except Exception as e:                                   # an old checkpoint format must not stop the rest
            print("C2", name, "FAILED", repr(e)[:300], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)

    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
