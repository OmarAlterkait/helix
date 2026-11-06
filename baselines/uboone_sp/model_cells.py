#!/usr/bin/env python3
"""Our denoiser's per-cell predictions on the ladder's cells, on CPU.

    scripts/helix_run.sh python baselines/uboone_sp/model_cells.py --events 260-387 \
        --out /pscratch/sd/o/oalter/helix_work/uboone_sp/model [--plane-event 300]

scripts/eval_denoise.py scores the model but writes only per-window maxima; the
per-cell charge resolution and the figures need its prediction per map row and
per window cell, so this replays eval_denoise's predict() (same checkpoint, same
tokens, same cell_inputs, cells no token covers -> 0) on CPU in fp32 (the GPU
run used bf16 autocast; score_ladder reports how closely this reproduces
dn.jsonl). ``--plane-event`` also predicts every token-covered cell of that
event (all planes) for the figures.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
FP_FRACS = (0.05, 0.25, 1.0)      # false-positive count thresholds, in units of Q0
CKPT = "/pscratch/sd/o/oalter/helix_work/denoise/ab_ft1024_win/best.pt"


def covered_cells_fn(cfg, R):
    """covered_cells(ce) for a tokenizer cfg (scripts/plot_denoise.covered_cells)."""
    from helix.model.tokenize import tick_of_tau

    def covered_cells(ce):
        """Every fine cell inside a token's footprint (scripts/plot_denoise.covered_cells)."""
        k = ce.band < cfg.n_bands
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
            out.append(R.fkey(np.broadcast_to(gb[:, None, None], shape)[sel],
                              np.broadcast_to(fw[:, :, None], shape)[sel],
                              np.broadcast_to(ft[:, None, :], shape)[sel]))
        return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)

    return covered_cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--truth", default="/pscratch/sd/o/oalter/helix_work/resolution/truth_v2")
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plane-event", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--fp-only", action="store_true",
                    help="predict every token-covered cell and store only the count of cells > 0.05 Q0 where "
                         "the truth is 0, per plane (the classical rungs count the same over full planes)")
    a = ap.parse_args()

    import h5py
    import torch
    torch.set_num_threads(a.threads)
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.model.tokenize import tick_of_tau
    from helix.probe import resolution as R

    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    hk = meta.get("head_kw") or {}
    presence = bool(hk.get("presence"))
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=hk)
    model.load_state_dict(ck["model"]); model.eval()
    Q0 = float(meta["q0"])
    ds = DenoiseEvents("", [], "", items=[]); cfg = ds.cfg
    del ck

    def predict(B, bl, keys):
        idx, aux = R.cell_inputs(keys, B["cell_key"], bl, cfg)
        ok = (idx >= 0).any(1)
        out = np.zeros(len(keys), np.float32)
        Bt = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        with torch.no_grad():
            feats = model.fm.encode(Bt)
            sel_all = np.nonzero(ok)[0]
            for s in range(0, len(sel_all), 65536):
                sel = sel_all[s:s + 65536]
                p = model.head(feats, torch.as_tensor(idx[sel]), torch.as_tensor(aux[sel]))
                out[sel] = (p[0] if presence else p).float().numpy()
        return out, ok

    covered_cells = covered_cells_fn(cfg, R)

    lo, hi = (int(x) for x in a.events.split("-"))
    for ev in range(lo, hi + 1):
        out = os.path.join(a.out, f"ev{ev:03d}.npz")
        if os.path.exists(out) and ev != a.plane_event:
            continue
        t0 = time.time()
        z = np.load(os.path.join(a.truth, f"ev{ev:03d}.npz"), allow_pickle=True)
        shard, event = str(z["shard"]), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        ce, B, bl = ds.tokens(shard, pos)
        if a.fp_only:
            from scripts.build_denoise_truth import event_cells
            SRC = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"
            tk, tq = event_cells(os.path.join(SRC, "hits", os.path.basename(os.path.dirname(shard)),
                                              ce.source_file.replace("_sensor_", "_hits_")), ce.event)
            cells = covered_cells(ce)
            pc, okc = predict(B, bl, cells)
            q = Q0 * np.expm1(np.maximum(pc, 0))
            empty = ~np.isin(cells, tk[tq > 0])
            cg = R.unkey(cells)[0]
            fp = np.array([[int(np.sum(okc & empty & (q > f * Q0) & (cg == g))) for g in range(6)] for f in FP_FRACS])
            np.savez(out, fp=fp, fp_fracs=np.array(FP_FRACS), n_covered=np.array([int(np.sum(okc & (cg == g))) for g in range(6)]))
            print(f"ev{ev:03d}: fp cells {fp.sum(1).tolist()} (> {FP_FRACS} Q0) of {int(okc.sum())} covered, {time.time() - t0:.0f}s", flush=True)
            continue
        pm, okm = predict(B, bl, z["mrows"])
        pw_, okw = predict(B, bl, z["wkey"]) if len(z["wkey"]) else (np.zeros(0, np.float32), np.zeros(0, bool))
        extra = {}
        if ev == a.plane_event:
            cells = covered_cells(ce)
            pc, okc = predict(B, bl, cells)
            extra = dict(cells=cells[okc], cells_p=pc[okc])
        np.savez(out, p_m=pm, ok_m=okm, p_w=pw_, ok_w=okw, Q0=Q0, **extra)
        print(f"ev{ev:03d}: {len(z['mrows'])} rows ({okm.mean():.3f} covered), {len(z['wkey'])} window cells, "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
