#!/usr/bin/env python3
"""A presence-headed denoiser's per-cell outputs for the ROI-level comparison, on CPU.

    scripts/helix_run.sh python baselines/uboone_sp/model_presence.py --events 260-387 \
        --checkpoint /pscratch/sd/o/oalter/helix_work/denoise/M4n_ft_full_near/final.pt \
        --out /pscratch/sd/o/oalter/helix_work/uboone_sp/model_m4n

Replays scripts/eval_denoise.py's predict() (same checkpoint, tokens, cell_inputs;
uncovered cells -> 0) with the encoder run ONCE per event, and stores:
  q_w / pres_w   charge (expm1(max(p,0)) x Q0) and presence probability at the
                 floor windows' cells (truth_v2 wkey)
  q_n / pres_n   the same at the near-activity windows' cells (truth_v2_near wkey)
  hist_*         over EVERY token-covered cell: per plane gid, histograms of the
                 presence probability (100 bins on [0, 1]) for cells with truth
                 charge and for truth-empty cells -- the model's 'ROI' purity at
                 any presence threshold.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
NEAR = "/pscratch/sd/o/oalter/helix_work/resolution/truth_v2_near"
SRC = "/global/cfs/cdirs/m5238/users/oalter/wire_test_00_00_02"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--truth", default="/pscratch/sd/o/oalter/helix_work/resolution/truth_v2")
    ap.add_argument("--events", default="260-387")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    import h5py
    import torch
    torch.set_num_threads(a.threads)
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.probe import resolution as R
    from scripts.build_denoise_truth import event_cells
    from model_cells import covered_cells_fn

    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    hk = meta.get("head_kw") or {}
    if not hk.get("presence"):
        raise SystemExit("this checkpoint has no presence head")
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=hk)
    model.load_state_dict(ck["model"]); model.eval()
    Q0 = float(meta["q0"])
    ds = DenoiseEvents("", [], "", items=[]); cfg = ds.cfg
    del ck
    covered_cells = covered_cells_fn(cfg, R)
    edges = np.linspace(0, 1, 101)

    lo, hi = (int(x) for x in a.events.split("-"))
    for ev in range(lo, hi + 1):
        out = os.path.join(a.out, f"ev{ev:03d}.npz")
        if os.path.exists(out):
            continue
        t0 = time.time()
        z = np.load(os.path.join(a.truth, f"ev{ev:03d}.npz"), allow_pickle=True)
        zn = np.load(os.path.join(NEAR, f"ev{ev:03d}.npz"), allow_pickle=True)
        shard, event = str(z["shard"]), int(z["event"])
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        ce, B, bl = ds.tokens(shard, pos)
        Bt = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]
        with torch.no_grad():
            feats = model.fm.encode(Bt)

            def predict(keys):
                q = np.zeros(len(keys), np.float32); pr = np.zeros(len(keys), np.float32)
                if not len(keys):
                    return q, pr, np.zeros(0, bool)
                idx, aux = R.cell_inputs(keys, B["cell_key"], bl, cfg)
                ok = (idx >= 0).any(1)
                sel_all = np.nonzero(ok)[0]
                for s in range(0, len(sel_all), 65536):
                    sel = sel_all[s:s + 65536]
                    p = model.head(feats, torch.as_tensor(idx[sel]), torch.as_tensor(aux[sel]))
                    q[sel] = Q0 * np.expm1(np.maximum(p[0].float().numpy(), 0))
                    pr[sel] = torch.sigmoid(p[1].float()).numpy()
                return q, pr, ok

            q_w, pres_w, _ = predict(z["wkey"])
            q_n, pres_n, _ = predict(zn["wkey"])
            cells = covered_cells(ce)
            _, pres_c, ok_c = predict(cells)
        tk, tq = event_cells(os.path.join(SRC, "hits", os.path.basename(os.path.dirname(shard)),
                                          ce.source_file.replace("_sensor_", "_hits_")), ce.event)
        has = np.isin(cells, tk[tq > 0])
        cg = R.unkey(cells)[0]
        hist_sig = np.stack([np.histogram(pres_c[ok_c & has & (cg == g)], edges)[0] for g in range(6)])
        hist_emp = np.stack([np.histogram(pres_c[ok_c & ~has & (cg == g)], edges)[0] for g in range(6)])
        np.savez(out, q_w=q_w, pres_w=pres_w, q_n=q_n, pres_n=pres_n, hist_sig=hist_sig, hist_emp=hist_emp,
                 edges=edges, Q0=Q0)
        print(f"ev{ev:03d}: {int(ok_c.sum())} covered cells, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
