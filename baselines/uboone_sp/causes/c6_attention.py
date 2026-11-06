#!/usr/bin/env python3
"""C6: the attention mechanism of M7dc's amplitude coupling, and a structural intervention.

The encoder (SerialFMModel) attends within fixed groups of a serialised order; layouts cycle per block:
plane-time (g=gp), DRIFT (time only, across all six planes and both volumes, g=gd), plane-wire (gp), DRIFT shifted.
(a) Attention, drift blocks only, reconstructed exactly (n1 -> qkv -> QK-norm -> RoPE as self_block): for bright query
    tokens (max |arcsinh value| > 4), the attention share on volume-0 keys of volume-1 queries (vs volume 0's share of
    the keys in their group), and on U(gid 0) keys of volume-0 V/Y queries; base vs the scaled variant (x1.05 on
    volume 0 / on gid 0). Competition predicts the share on the brightened keys rises.
(b) Intervention: drift layouts sorted by (volume, time) instead of time -- the weights untouched -- then M7dc's
    bright-cell accuracy and its cross-volume coupling (volume 0 x1.05 -> volume-1 cells), vs the original layouts.

    CUDA_VISIBLE_DEVICES=3 scripts/helix_run.sh python baselines/uboone_sp/causes/c6_attention.py --out <json>
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
CT = "/global/cfs/cdirs/m5238/users/oalter/denoise_truth_complete"
TRUTH_V2 = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2"
CKPT = f"{R}/denoise/M7dc_ft_full_dec_cond_ct25/final.pt"
Q0 = 2043.2578108909702


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="260-267")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import h5py
    import torch
    import types
    from helix.core.coeff_io import read_coeff_event
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.model.layers import apply_rope_tables, qk_normed, rope_tables
    from helix.model.tokenize import assemble, to_fm
    from helix.probe import resolution as R_
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = ck["meta"]; ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=meta["head_kw"])
    model.load_state_dict(ck["model"]); model.to(dev).eval(); del ck
    fm = model.fm
    print("encoder", type(fm).__name__, "blocks", len(fm.enc), "gp", fm.gp, "gd", fm.gd, "rope_split", fm.rope_split,
          "varlen", fm.varlen, flush=True)
    ds = DenoiseEvents(CORPUS, [], CT, items=[]); cfg = ds.cfg
    orig_layouts = fm._layouts

    def volume_layouts(plane, t, wire, depth=None):
        """drift layouts sorted by (volume, time): attention cannot cross volumes there."""
        L = orig_layouts(plane, t, wire, depth)
        vol = (plane // 3).double()
        o_t = torch.argsort(vol * 1e9 + t.double()); o_ts = torch.roll(o_t, fm.gd // 2)
        out = []
        for i, (o, g, uw) in enumerate(L):
            out.append((o_t, g, uw) if i % 4 == 1 else (o_ts, g, uw) if i % 4 == 3 else (o, g, uw))
        return out

    def tokens(ce, c, scale_mask):
        v = ce.value.astype(np.float64) * (1.0 if scale_mask is None else np.where(scale_mask, 1.05, 1.0))
        B = to_fm(assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, v.astype(np.float32), gids=c["gids"], n_wires=c["n_wires"],
                           band_lengths=c["band_lengths"], norm_sigma=c["norm_sigma"], cfg=cfg))
        Bt = {k: torch.as_tensor(x).to(dev) if isinstance(x, np.ndarray) else x for k, x in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        return B, Bt

    @torch.no_grad()
    def drift_attention(Bt):
        """per drift block: (share of volume-0 keys, share on U keys) per query token, natural order."""
        plane, t, wire = Bt["plane_id"], Bt["t_phys"], Bt["wire_pos"]
        at, aw = fm._angles(Bt)
        L = len(fm.enc)
        X = fm.encode_layers(Bt, list(range(1, L + 1)))
        x_in = {0: fm._emb(Bt).float()}
        for i in range(1, L):
            x_in[i] = X[i].float()
        lays = orig_layouts(plane, t, wire)
        vol0 = (plane // 3 == 0); isU = (plane == 0)
        res = {}
        for i, blk in enumerate(fm.enc):
            o, g, uw = lays[i % len(lays)]
            if g != fm.gd:
                continue
            T = o.numel()
            xo = x_in[i][o]
            hh = blk.n1(xo)
            q, k, _ = blk.qkv(hh).chunk(3, -1)
            q, k = qk_normed(blk, q.view(T, blk.h, blk.hd), k.view(T, blk.h, blk.hd))
            cos, sin = rope_tables(at[o], aw[o] if uw else None, q.dtype)
            q, k = apply_rope_tables(q, cos, sin), apply_rope_tables(k, cos, sin)
            scale = blk.attn_scale if blk.attn_scale is not None else blk.hd ** -0.5
            m_v0 = torch.zeros(T, device=dev); m_u = torch.zeros(T, device=dev); s_v0 = torch.zeros(T, device=dev)
            v0o, uo = vol0[o].float(), isU[o].float()
            for s in range(0, T, g):
                e = min(s + g, T)
                qq, kk = q[s:e].float().permute(1, 0, 2), k[s:e].float().permute(1, 0, 2)
                p = (qq @ kk.transpose(1, 2) * scale).softmax(-1).mean(0)                 # (n, n) head-averaged
                m_v0[s:e] = p @ v0o[s:e]; m_u[s:e] = p @ uo[s:e]
                s_v0[s:e] = v0o[s:e].mean()
            inv = torch.empty_like(o); inv[o] = torch.arange(T, device=dev)
            res[i] = (m_v0[inv].cpu().numpy(), m_u[inv].cpu().numpy(), s_v0[inv].cpu().numpy())
        return res

    @torch.no_grad()
    def predict(ce, c, keys, scale_mask=None):
        B, Bt = tokens(ce, c, scale_mask)
        idx, aux = R_.cell_inputs(keys, B["cell_key"], c["band_lengths"], cfg)
        ok = (idx >= 0).any(1)
        y = np.full(len(keys), np.nan)
        feats = fm.encode(Bt)
        nbr = torch.as_tensor(R_.cell_neighbors(keys[ok], B["cell_key"], c["band_lengths"], cfg)).to(dev)
        o = model.head(feats, torch.as_tensor(idx[ok]).to(dev), torch.as_tensor(aux[ok]).to(dev), nbr=nbr, B=Bt)
        y[ok] = o[0].float().cpu().numpy()
        return y

    att = {"v1_on_v0_base": [], "v1_on_v0_scaled": [], "v1_keyshare_v0": [], "vy_on_u_base": [], "vy_on_u_scaled": []}
    pred = {m: {k: [] for k in ("g", "T", "y0", "yv")} for m in ("orig", "volume_drift")}
    lo, hi = map(int, a.events.split("-"))
    for ev in range(lo, hi + 1):
        z = np.load(os.path.join(TRUTH_V2, f"ev{ev:03d}.npz"), allow_pickle=True)
        run = os.path.basename(os.path.dirname(str(z["shard"])))
        path = os.path.join(CORPUS, run, os.path.basename(str(z["shard"])))
        with h5py.File(path, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], int(z["event"])))
            c = {k: fh["config"][k][:] for k in ("gids", "n_wires", "band_lengths", "norm_sigma")}
        ce = read_coeff_event(path, pos)
        # (a) attention in the original layouts
        B0, Bt0 = tokens(ce, c, None)
        gid = np.asarray(B0["plane_id"]); bright = np.abs(np.asarray(B0["inp"])).max(1) > 4
        A0 = drift_attention(Bt0)
        _, BtV = tokens(ce, c, (ce.plane_gid // 3) == 0); AV = drift_attention(BtV)
        _, BtU = tokens(ce, c, ce.plane_gid == 0); AU = drift_attention(BtU)
        q1 = bright & (gid // 3 == 1); qvy = bright & np.isin(gid, [1, 2])
        for i in A0:
            att["v1_on_v0_base"].append(A0[i][0][q1]); att["v1_on_v0_scaled"].append(AV[i][0][q1])
            att["v1_keyshare_v0"].append(A0[i][2][q1])
            att["vy_on_u_base"].append(A0[i][1][qvy]); att["vy_on_u_scaled"].append(AU[i][1][qvy])
        # (b) intervention: volume-aware drift layouts
        tkeys, tq, _ = ds._truth(run, path, pos)
        sel = tq >= 16e3
        keys, T = tkeys[sel].astype(np.int64), tq[sel]
        g_, _, _ = R_.unkey(keys)
        for mode in ("orig", "volume_drift"):
            fm._layouts = orig_layouts if mode == "orig" else volume_layouts
            y0 = predict(ce, c, keys); yv = predict(ce, c, keys, (ce.plane_gid // 3) == 0)
            for k, v in (("g", g_), ("T", T), ("y0", y0), ("yv", yv)):
                pred[mode][k].append(v)
        fm._layouts = orig_layouts
        print(f"ev{ev:03d} done", flush=True)
    st = lambda x: (lambda a16, a50, a84: dict(med=float(a50), mean=float(np.mean(x)), hw68=float((a84 - a16) / 2), n=int(len(x))))(*np.percentile(x, [16, 50, 84]))
    out = {"attention": {}, "intervention": {}}
    A = {k: np.concatenate(v) for k, v in att.items()}
    out["attention"]["v1_queries_share_on_vol0_keys"] = dict(base=st(A["v1_on_v0_base"]), vol0_scaled=st(A["v1_on_v0_scaled"]),
                                                             key_share_of_vol0_in_group=st(A["v1_keyshare_v0"]),
                                                             change=st(A["v1_on_v0_scaled"] - A["v1_on_v0_base"]))
    out["attention"]["vy_queries_share_on_U_keys"] = dict(base=st(A["vy_on_u_base"]), U_scaled=st(A["vy_on_u_scaled"]),
                                                          change=st(A["vy_on_u_scaled"] - A["vy_on_u_base"]))
    for mode, P in pred.items():
        D = {k: np.concatenate(v) for k, v in P.items()}
        yT = np.log1p(D["T"] / Q0); p = D["g"] % 3
        q0 = Q0 * np.expm1(np.maximum(D["y0"], 0)); lin = np.log1p(q0 * 1.05 / Q0) - np.log1p(q0 / Q0)
        s = (D["yv"] - D["y0"]) / lin
        r = dict(coupling_vol0_to_vol1=st(s[np.isfinite(s) & (D["g"] >= 3)]), inside_vol0=st(s[np.isfinite(s) & (D["g"] < 3)]))
        for pl in range(3):
            for lo_, hi_ in ((16e3, 64e3), (64e3, np.inf)):
                m = (p == pl) & (D["T"] >= lo_) & (D["T"] < hi_) & np.isfinite(D["y0"])
                r[f"acc_{'UVY'[pl]}_{lo_:.0f}"] = st(D["y0"][m] - yT[m])
        out["intervention"][mode] = r
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: {kk: (vv["med"], vv["mean"]) if isinstance(vv, dict) and "med" in vv else vv for kk, vv in v.items()}
                      for k, v in out["attention"].items()}, indent=1))
    for mode, r in out["intervention"].items():
        print(mode, {k: (round(v["med"], 4), round(v["hw68"], 4)) for k, v in r.items()})


if __name__ == "__main__":
    main()
