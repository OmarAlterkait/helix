"""Stage 2: probe a checkpoint against the per-pixel truth artifact.

    python scripts/run_probe.py --checkpoint /path/fm.pt \
        --corpus /sdf/data/neutrino/omara/coeff_tpc/run_0027575715 \
        --tag m113 --out probe_results.jsonl

Verifies the truth artifact against the split it claims to describe, extracts
frozen features at one layer, joins them to truth through the tokenizer's own
``pixel_cells``, and fits both probes with event-grouped folds and early
stopping.

Everything that changes a number is echoed into every results row — the weights
used (EMA or raw), the layer, ``cell_t``, ``qtot_min``, ``dom_threshold``, the
fold and epoch budget. Two ``fisher_r`` values from different definitions must
never be comparable by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_truth(path, corpus, strict=True):
    """Read the artifact and check it still describes this corpus's split."""
    import h5py

    with h5py.File(path, "r") as f:
        c = f["config"]
        cfg = {k: c.attrs[k] for k in c.attrs}
        aw = c["alongwire"][:]
        pix = {k: f["pix"][k][:] for k in ("gid", "wire", "tick", "qtot", "ftop", "b1")}
        offs = f["pix"]["event_offset"][:]
        ident = [(r.decode() if isinstance(r, bytes) else str(r),
                  s.decode() if isinstance(s, bytes) else str(s), int(e))
                 for r, s, e in zip(f["ident"]["run"][:], f["ident"]["source_file"][:],
                                    f["ident"]["event"][:])]

    hj = os.path.join(corpus, "holdout.json")
    want_hold = hashlib.sha256(open(hj, "rb").read()).hexdigest()
    got_hold = str(cfg.get("holdout_json_sha256", ""))
    ident_txt = "\n".join(sorted(f"{r}/{s}/{e}" for r, s, e in ident))
    got_ident = hashlib.sha256(ident_txt.encode()).hexdigest()

    problems = []
    if got_hold != want_hold:
        problems.append(
            f"holdout.json has changed since the truth was dumped "
            f"({got_hold[:12]} != {want_hold[:12]})")
    if got_ident != str(cfg.get("corpus_ident_sha256", "")):
        problems.append("the artifact's own event list does not match its recorded digest")
    if problems and strict:
        raise SystemExit("stale truth artifact:\n  - " + "\n  - ".join(problems) +
                         "\n  re-run scripts/dump_probe_truth.py")
    return cfg, aw, pix, offs, ident, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--truth", default=None, help="default <corpus>/truth/probe_truth_probe.h5")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--dataset-name", default="sim_wire")
    ap.add_argument("--out", default="probe_results.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--probes", default="mlp,triangulate")
    ap.add_argument("--allow-stale", action="store_true")
    a = ap.parse_args(argv)

    import h5py
    import torch
    from helix.core.coeff_io import read_coeff_event
    from helix.model.tokenize import PatchConfig, assemble, to_fm, pixel_cells
    from helix.model.checkpoint import patch_config_from_checkpoint
    from helix.probe.alongwire import u_of
    from helix.probe.designs import mlp_designs, triangulate_designs
    from helix.probe.features import (features_at_layer, gather_cell_features,
                                      load_probe_model)
    from helix.probe.fit import fit_probe
    from helix.probe.metrics import fisher_r
    from helix.probe.patches import patch_rows

    corpus = os.path.abspath(a.corpus)
    truth_path = a.truth or os.path.join(corpus, "truth", "probe_truth_probe.h5")
    cfg, aw, pix, offs, ident, stale = _load_truth(truth_path, corpus,
                                                   strict=not a.allow_stale)
    if stale:
        print("WARNING (--allow-stale):", "; ".join(stale), flush=True)

    # The tokenizer geometry comes from the CHECKPOINT, never from the default.
    pcfg = patch_config_from_checkpoint(a.checkpoint) or PatchConfig()
    print(f"tokenizer: cell_t={pcfg.cell_t} pw={pcfg.pw} pt={pcfg.pt}", flush=True)

    n_ev = len(ident) if not a.max_events else min(a.max_events, len(ident))
    print(f"{n_ev} probe events from {os.path.basename(truth_path)}", flush=True)

    model_t, meta_t = load_probe_model(a.checkpoint, weights=a.weights)
    model_r, _ = load_probe_model(a.checkpoint, random_init=True)
    if meta_t.get("warning"):
        print("WARNING:", meta_t["warning"], flush=True)

    packs = []
    for i in range(n_ev):
        run, src, ev = ident[i]
        tag = src.replace("sim_wire_sensor_", "").replace(".h5", "")
        shard = os.path.join(corpus, f"{a.dataset_name}_coeff_{tag}.h5")
        with h5py.File(shard, "r") as f:
            c = f["config"]
            gids, nw = c["gids"][:], c["n_wires"][:]
            bl, ns = c["band_lengths"][:], c["norm_sigma"][:]
        ce = read_coeff_event(shard, ev)
        tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value,
                       gids=gids, n_wires=nw, band_lengths=bl, norm_sigma=ns,
                       cfg=pcfg)
        B = to_fm(tok)
        Bt = {k: (torch.as_tensor(v).to(next(model_t.parameters()).device)
                  if isinstance(v, np.ndarray) else v) for k, v in B.items()}

        # cell key -> row index in the token set, so pixels can address features
        keys = tok["cell_key"]          # sorted, in cell-index order
        lo, hi = offs[i], offs[i + 1]
        P = {k: v[lo:hi] for k, v in pix.items()}
        pc = pixel_cells(P["gid"], P["wire"], P["tick"], bl, pcfg)
        rows = np.searchsorted(keys, pc)
        rows = np.where((rows < len(keys)) & (keys[np.clip(rows, 0, len(keys) - 1)] == pc),
                        rows, -1)

        u = u_of(P["gid"].astype(np.int64), P["b1"], aw)
        pr = patch_rows(rows, P["gid"].astype(np.int64), P["wire"], P["tick"],
                        P["qtot"], P["ftop"], u,
                        dom_threshold=float(cfg.get("dom_threshold", 0.5)))
        if not len(pr["y"]):
            continue

        ft = features_at_layer(model_t, Bt, a.layer)
        fr = features_at_layer(model_r, Bt, a.layer)
        pr["Xtr"] = gather_cell_features(ft, pr["cells"]).cpu().numpy()
        pr["Xrn"] = gather_cell_features(fr, pr["cells"]).cpu().numpy()
        pr["Xraw"] = gather_cell_features(
            torch.as_tensor(B["inp"]).float().to(ft.device), pr["cells"]).cpu().numpy()
        pr["event"] = np.full(len(pr["y"]), i)
        pr["tick"] = np.zeros(len(pr["y"]))     # patch mean tick, from geo col 7
        pr["tick"] = pr["geo"][:, 7] * 4321.0
        pr["wire"] = pr["geo"][:, 6] * 2000.0
        packs.append(pr)
        del ft, fr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or i + 1 == n_ev:
            print(f"  {i+1}/{n_ev} events, {sum(len(p['y']) for p in packs):,} patches",
                  flush=True)

    if not packs:
        raise SystemExit("no patches survived — check the truth/corpus pairing")
    cat = lambda k: np.concatenate([p[k] for p in packs])
    y, event, plane = cat("y"), cat("event"), cat("plane")
    geo = cat("geo")
    arms = {"trained": cat("Xtr"), "random": cat("Xrn"), "raw": cat("Xraw")}
    print(f"total {len(y):,} patches over {len(np.unique(event))} events, "
          f"feature dim {arms['trained'].shape[1]}", flush=True)

    base = dict(tag=a.tag, checkpoint=os.path.abspath(a.checkpoint),
                layer=a.layer, weights=meta_t["weights"],
                cell_t=pcfg.cell_t, pw=pcfg.pw, pt=pcfg.pt,
                qtot_min=float(cfg.get("qtot_min", -1)),
                dom_threshold=float(cfg.get("dom_threshold", -1)),
                folds=a.folds, epochs=a.epochs, seeds=a.seeds,
                n_patch=int(len(y)), n_events=int(len(np.unique(event))),
                truth=os.path.abspath(truth_path), when=int(time.time()))

    seeds = tuple(range(a.seeds))
    out_rows = []
    wanted = [p.strip() for p in a.probes.split(",") if p.strip()]

    if "mlp" in wanted:
        designs = mlp_designs(geo, arms)
        row = dict(base, probe="mlp")
        for name, X in designs.items():
            oof, info = fit_probe(X, y, event, plane, n_folds=a.folds,
                                  epochs=a.epochs, seeds=seeds)
            r, rs, minfo = fisher_r(y, oof, event, plane)
            row[name] = dict(fisher_r=round(r, 4), std=round(float(rs.std()), 4),
                             n_groups=minfo["n_groups"],
                             stop_epoch=round(info["mean_stop_epoch"], 1))
            print(f"  [mlp] {name:8s} fisher_r={r:+.4f}", flush=True)
        if "geo" in row and "trained" in row:
            row["d_over_geo_r"] = round(row["trained"]["fisher_r"] - row["geo"]["fisher_r"], 4)
        out_rows.append(row)

    if "triangulate" in wanted:
        d = triangulate_designs(geo, arms["trained"], plane, cat("tick"),
                                   cat("wire"), event)
        row = dict(base, probe="triangulate")
        for name, X in d.items():
            oof, info = fit_probe(X, y, event, plane, n_folds=a.folds,
                                  epochs=a.epochs, seeds=seeds)
            r, rs, minfo = fisher_r(y, oof, event, plane)
            row[name] = dict(fisher_r=round(r, 4), std=round(float(rs.std()), 4),
                             n_groups=minfo["n_groups"],
                             stop_epoch=round(info["mean_stop_epoch"], 1))
            print(f"  [tri] {name:8s} fisher_r={r:+.4f}", flush=True)
        out_rows.append(row)

    with open(a.out, "a") as f:
        for r in out_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"wrote {len(out_rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
