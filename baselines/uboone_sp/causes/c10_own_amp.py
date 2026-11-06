#!/usr/bin/env python3
"""L2 (results/m7_tests_plan.md): does each encoder layer still carry every token's OWN coefficient amplitudes, and
to what precision? A nonlinear probe (MLP d -> 512 -> 512 -> n_slot) per layer predicts the token's arcsinh slot
values (loss on occupied slots), fitted on calibration events, scored on test events: hw68 of the residual on
bright occupied slots (|value| > 4) per band. Pretrained FM and M7dc's fine-tuned encoder, layers 0 (embedding)..12.
Bin-quantisation reference (uniform within a bin): hw68 = 0.34 w = A4 0.028, D4 0.024, D3 0.021, D2 0.017.

    CUDA_VISIBLE_DEVICES=k scripts/helix_run.sh python baselines/uboone_sp/causes/c10_own_amp.py --encoder pretrained --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
R = "/global/cfs/cdirs/m5238/users/oalter/helix_runs"
CORPUS = "/global/cfs/cdirs/m5238/users/oalter/ost61/corpus"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
ART = "/pscratch/sd/o/oalter/helix_work/probe/artifact_nz_pw16_s1"
CKPT = f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt"
LAYERS = (0, 1, 2, 3, 4, 6, 8, 10, 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["pretrained", "m7dc"], required=True)
    ap.add_argument("--cal", default="0-23")
    ap.add_argument("--test", default="260-271")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.model.tokenize import PatchConfig, assemble, to_fm
    dev = torch.device("cuda")
    torch.manual_seed(0)
    if a.encoder == "pretrained":
        from helix.model.artifact import load
        from helix.model.fm import build_fm
        art = load(ART); arch = dict(art.arch); arch["compile_blocks"] = False
        fm = build_fm(arch); fm.load_state_dict(art.state_dict)
    else:
        from helix.model.denoise import build_denoise
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
        m = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta["head_kw"]); m.load_state_dict(ck["model"])
        fm = m.fm
    fm.to(dev).eval()
    cfg = PatchConfig(cell_t="grid_center")
    L = len(fm.enc)

    @torch.no_grad()
    def feats(ev):
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        run = os.path.basename(os.path.dirname(str(z["shard"])))
        path = os.path.join(CORPUS, run, os.path.basename(str(z["shard"])))
        if not os.path.exists(path):
            return None
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
            c = {k: fh["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        ce = read_coeff_event(path, pos)
        B = to_fm(assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, gids=c["gids"], n_wires=c["n_wires"],
                           band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg))
        Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        X = fm.encode_layers(Bt, [k for k in LAYERS if k > 0])
        X[0] = fm._emb(Bt)
        inp = np.asarray(B["inp"]); occ = np.asarray(B["occ"]) > 0
        keep = (np.abs(inp) * occ).max(1) > 4
        kt = torch.as_tensor(keep).to(dev)
        return ({k: X[k][kt].float() for k in LAYERS}, torch.as_tensor(inp[keep]).to(dev), torch.as_tensor(occ[keep]).to(dev),
                np.asarray(B["band_id"])[keep])

    def gather(rng_):
        lo, hi = map(int, rng_.split("-"))
        F = [f for f in (feats(ev) for ev in range(lo, hi + 1) if ev != 41) if f is not None]
        X = {k: torch.cat([f[0][k] for f in F]) for k in LAYERS}
        return X, torch.cat([f[1] for f in F]), torch.cat([f[2] for f in F]), np.concatenate([f[3] for f in F])

    Xc, Yc, Oc, _ = gather(a.cal)
    Xt, Yt, Ot, Bt_band = gather(a.test)
    print(f"cal tokens {len(Yc)}, test tokens {len(Yt)}", flush=True)
    out = {}
    for k in LAYERS:
        x, xt = Xc[k], Xt[k]
        mu, sd = x.mean(0), x.std(0).clamp(min=1e-6)
        d = x.shape[1]; ns = Yc.shape[1]
        net = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.GELU(), torch.nn.Linear(512, 512), torch.nn.GELU(),
                                  torch.nn.Linear(512, ns)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        n = len(x)
        for s in range(a.steps):
            i = torch.randint(0, n, (4096,), device=dev)
            p = net((x[i] - mu) / sd)
            w = Oc[i].float()
            loss = (((p - Yc[i]) ** 2) * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            P = torch.cat([net((xt[j:j + 8192] - mu) / sd) for j in range(0, len(xt), 8192)])
        E = (P - Yt).cpu().numpy(); inp = Yt.cpu().numpy(); occ = Ot.cpu().numpy()
        r = {}
        for b in range(4):
            m = occ & (np.abs(inp) > 4) & (Bt_band == b)[:, None]
            e = E[m]
            a16, a84 = np.percentile(e, [16, 84])
            r[f"band{b}"] = dict(hw68=float((a84 - a16) / 2), med=float(np.median(e)), n=int(len(e)))
        out[k] = r
        print(a.encoder, "layer", k, {b: round(v["hw68"], 4) for b, v in r.items()}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
