#!/usr/bin/env python3
"""Score a trained denoiser on the floor evaluation, exactly as the probe is scored.

    python scripts/eval_denoise.py --checkpoint <out>/best.pt --truth <truth_v2> --tag T --out res.jsonl

Same test events (truth files ``ev260``-``ev387`` of the probe split), same cells,
same windows, same ``Q0`` and the same :func:`helix.probe.resolution.scalars` as
scripts/eval_resolution.py -- only the per-cell prediction comes from the trained
model instead of a probe on frozen features, so the numbers are directly
comparable with every probe result. Cells no token covers are predicted 0.
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
    ap.add_argument("--truth", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-events", type=int, default=260, help="probe-split events NOT scored (the probe's training set)")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()

    import torch
    from helix.data.denoise import DenoiseEvents
    from helix.model.denoise import build_denoise
    from helix.probe import resolution as R

    dev = torch.device("cuda")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    ov = dict(meta["overrides"]); ov["compile_blocks"] = False
    hk = meta.get("head_kw") or {}
    presence = bool(hk.get("presence"))
    model = build_denoise(meta["arch"], None, overrides=ov, head_kw=hk)
    model.load_state_dict(ck["model"]); model.to(dev).eval()
    q0 = float(meta["q0"])
    ds = DenoiseEvents("", [], "", items=[])                  # tokenizer + cfg only
    cfg = ds.cfg

    files = sorted(glob.glob(os.path.join(a.truth, "ev*.npz")))
    qs = np.concatenate([np.load(f, allow_pickle=True)["mq"] for f in files if int(os.path.basename(f)[2:5]) < a.train_events])
    Q0 = float(np.median(qs[qs > 0]))
    if abs(Q0 - q0) > 1e-6 * Q0:
        raise SystemExit(f"model trained with Q0={q0}, this truth has Q0={Q0}: predictions are on another scale")

    P, PR, Y, E, PL, WQ, WID, WP, WPR, META = [], [], [], [], [], [], [], [], [], []
    for ei, f in enumerate(files):
        if int(os.path.basename(f)[2:5]) < a.train_events:
            continue
        z = np.load(f, allow_pickle=True)
        shard, event = str(z["shard"]), int(z["event"])
        import h5py
        with h5py.File(shard, "r") as fh:
            pos = int(np.searchsorted(fh["ident"]["event"][:], event))
        _, B, bl = ds.tokens(shard, pos)
        Bt = {k: torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v for k, v in B.items()}
        Bt["n_cells"] = Bt["plane_id"].shape[0]

        def predict(keys):
            idx, aux = R.cell_inputs(keys, B["cell_key"], bl, cfg)
            ok = (idx >= 0).any(1)
            out = np.zeros(len(keys), np.float32)
            pres = np.zeros(len(keys), np.float32)
            if ok.any():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(Bt, torch.as_tensor(idx[ok]).to(dev), torch.as_tensor(aux[ok]).to(dev))
                if presence:
                    pres[ok] = torch.sigmoid(p[1].float()).cpu().numpy(); p = p[0]
                out[ok] = p.float().cpu().numpy()
            return out, ok, pres

        pm, okm, prm = predict(z["mrows"])
        y = np.log1p(z["mq"] / Q0).astype(np.float32)
        P.append(pm[okm]); PR.append(prm[okm]); Y.append(y[okm]); E.append(np.full(okm.sum(), ei))
        PL.append((R.unkey(z["mrows"][okm])[0] % 3).astype(np.int8))
        if len(z["wkey"]):
            pw_, _, pr_ = predict(z["wkey"])                  # uncovered cells stay 0
            base = len(META)
            mm = json.loads(str(z["meta"]))
            for m_ in mm:
                m_["ev"] = ei
            META += mm; WQ.append(z["wq"]); WID.append(z["wid"] + base); WP.append(pw_); WPR.append(pr_)
        if ei % 32 == 0:
            print(f"  event {ei}", flush=True)

    y, e, pl = map(np.concatenate, (Y, E, PL))
    wq, wid = map(np.concatenate, (WQ, WID))
    starts = np.searchsorted(wid, np.arange(len(META))); ends = np.append(starts[1:], len(wid))
    toff = {i: cfg.toff[i] for i in range(3)}
    q = np.expm1(y) * Q0; q30 = np.quantile(q[q > 0], 0.3)        # as eval_resolution: truth in charge units

    def score(tag, p, wscore):
        """Every metric for one per-cell score ``p`` (map rows) and window cell
        score ``wscore``: charge (expm1(max(p,0))*Q0, exactly eval_resolution's
        window input) or presence probability."""
        W = R.window_stats(wscore, wq, META, starts, ends, cfg.pw, cfg.pt, cfg.delta[0], toff)
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
        rng = np.random.default_rng(0); boots = []
        for _ in range(a.boot):
            sm = rng.choice(evs, len(evs), replace=True)
            idx = np.concatenate([rows_of[v] for v in sm])
            boots.append(R.scalars(p[idx], q[idx], [z_ for v in sm for z_ in z_of[v]], [w for v in sm for w in wins_of[v]], q30))
        r = {"val_mse": float(np.mean((np.concatenate(P) - y) ** 2))}
        for k, v in point.items():
            bv = np.array([b_[k] for b_ in boots], float)
            lo, hi = np.nanpercentile(bv, [16, 84]) if np.isfinite(bv).any() else (np.nan, np.nan)
            r[k] = [round(v, 4), round(float(lo), 4), round(float(hi), 4)]
        res = dict(tag=tag, checkpoint=a.checkpoint, step=ck.get("step"), Q0=Q0, n_windows=len(META),
                   train_events=meta["args"].get("n_events"), init=meta["args"].get("init"), trained=r)
        with open(a.out.replace(".jsonl", f"_{tag}_trained_windows.json"), "w") as fh:
            json.dump([dict({k: w[k] for k in ("kind", "score", "eb", "dip", "edge", "sep") if k in w},
                            ev=m_.get("ev"), g=m_.get("g"), w0=m_.get("w0"), t0=m_.get("t0"))
                       for w, m_ in zip(W, META)], fh, default=float)       # where each window is, for diagnosis
        with open(a.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(tag, json.dumps({k: r[k] for k in ("map_r", "faint_auc", "floor_eff1pct_0.1-0.2", "floor_auc_0-0.1")}), flush=True)

    wp = np.concatenate(WP)
    score(a.tag, np.concatenate(P), np.expm1(np.maximum(wp, 0)) * Q0)
    if presence:                      # the same cells scored by "is there charge here"
        score(a.tag + "_pres", np.concatenate(PR), np.concatenate(WPR))


if __name__ == "__main__":
    main()
