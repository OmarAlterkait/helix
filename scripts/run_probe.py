"""Stage 2: probe a checkpoint against the per-pixel truth artifact.

    python scripts/run_probe.py --checkpoint /path/fm.pt \
        --corpus /sdf/data/neutrino/omara/coeff_tpc/run_0027575715 \
        --tag m113 --out probe_results.jsonl

Verifies the truth artifact against the split it claims to describe, extracts
frozen features at one layer, joins them to truth through the tokenizer's own
``pixel_cells``, and fits both probes with event-grouped folds and early
stopping.

Everything that changes a number is echoed into every results row — the weights
used (EMA or raw) and their content digest, the layer, ``cell_t``, ``qtot_min``,
``dom_threshold``, the fold and epoch budget, which dataset was opened
(``dataset_name``), which split (``holdout_sha256``, ``corpus_ident_sha256``),
and which code produced it (``code``). Two ``fisher_r`` values from different
definitions must never be comparable by accident.
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


def _provenance_or_none():
    """Commit/branch/dirty for helix, pimm and pimm-data, or None.

    Never fatal: a results row that exists without provenance is worth more than
    no row at all, and this runs inside a long GPU job.
    """
    try:
        from helix.integrations._bootstrap import provenance
        return provenance()
    except Exception:
        return None


def _weights_digest_or_none(model):
    """`_weights_digest`, but never fatal -- same contract as `_provenance_or_none`.

    It is called while building the results row, which happens AFTER the whole
    per-event GPU loop and BEFORE the first `_emit`. Unguarded, any exception
    there discards a completed extraction -- the exact failure `_emit`'s own
    docstring was written about ("an OOM in triangulate discarded a complete
    four-arm mlp result that had already run"). Its sibling on the very next
    field is guarded; this one was not.

    The known edge cases are already handled inside `_weights_digest` --
    `flatten().view(torch.uint8)` copes with bfloat16 and 0-dim buffers, checked
    against live modules -- so this catches what is left: a CUDA transfer error,
    `state_dict()` itself raising, an exotic buffer dtype. Rare, and cheap to
    survive: a row with a null digest beats no row.
    """
    try:
        return _weights_digest(model)
    except Exception:
        return None


def _position_of(shard, event_id):
    """Position of ``event_id`` within a shard — NOT the id itself.

    ``read_coeff_event`` slices by ``event_offset``, i.e. by POSITION. Ids and
    positions coincide on every shard whose source events are contiguous, and
    diverge on the ones that are not: ``sim_wire_sensor_0065.h5`` is missing
    ``event_167``, so id 176 sits at position 175 there. Passing an id straight
    in reads the NEXT event, silently, on that shard only.

    Caught by the coverage check, which scored 0.383 where a matched event
    scores 0.999 — the guard pointing at the consumer rather than the data.
    """
    import h5py
    with h5py.File(shard, "r") as f:
        ids = f["ident"]["event"][:]
    pos = int(np.searchsorted(ids, event_id))
    if pos >= len(ids) or int(ids[pos]) != int(event_id):
        raise SystemExit(
            f"{shard}: no event with id {event_id} (shard holds "
            f"{len(ids)} events, {ids.min()}..{ids.max()})")
    return pos


def _weights_digest(model):
    """blake2b over the scored weights — the only field that identifies WHICH
    weights produced a row.

    `weights_source` is a path the export recorded and `weights_are_ema` is a
    substring test on a filename; neither survives a file being moved, renamed,
    or re-exported.

    Same SHAPE as tools/convert_fm_ckpt.py:_digest -- blake2b over sorted keys
    and contiguous CPU bytes -- but NOT the same value, deliberately, on two
    counts. The dtype is hashed, because two tensors with identical bytes under
    different dtypes are different weights. And `num_batches_tracked` /
    `n_averaged` are excluded, because they are step counters registered as
    persistent buffers: including them would make the digest partly a function
    of how long training ran. Compare these digests to each other, never to a
    convert_fm_ckpt one.

    Bytes go through `flatten().view(torch.uint8)` rather than `.numpy()`:
    `.numpy()` raises on bfloat16, and `view(torch.uint8)` raises on a 0-dim
    tensor of a different element size, so a scalar buffer would crash the row.
    """
    import hashlib
    import torch
    h = hashlib.blake2b(digest_size=16)
    sd = model.state_dict()
    for k in sorted(sd):
        if k.endswith(("num_batches_tracked", "n_averaged")):
            continue
        v = sd[k]
        if not torch.is_tensor(v):
            continue
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(v.detach().cpu().contiguous().flatten()
                 .view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _emit(path, row):
    """Append one completed probe row immediately.

    Both probes used to be written only after BOTH finished, so an OOM in
    triangulate discarded a complete four-arm mlp result that had already run.
    """
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"  -> appended {row['probe']} to {path}", flush=True)


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
    ap.add_argument("--random-seed", type=int, default=0,
                    help="seeds the random-init control. It was unseeded, so the "
                         "null was a different network in every process and two "
                         "arms of an A/B were scored against two different "
                         "controls with nothing recording it.")
    ap.add_argument("--probes", default="mlp,triangulate")
    ap.add_argument("--allow-stale", action="store_true")
    ap.add_argument("--cell-t", default=None, choices=("grid_center", "centroid"),
                    help="tokenizer cell_t, REQUIRED when the checkpoint records no "
                         "tokenizer. helix configs train grid_center.")
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
    pcfg = patch_config_from_checkpoint(a.checkpoint, cell_t=a.cell_t)
    src = "checkpoint" if pcfg is not None else "--cell-t"
    if pcfg is None:
        if getattr(a, "cell_t", None) is None:
            raise SystemExit(
                f"{a.checkpoint} records no tokenizer, and --cell-t was not given.\n"
                "Refusing to guess. The old fallback was `... or PatchConfig()`, which\n"
                "silently chose cell_t='centroid' while every helix config trains\n"
                "'grid_center' -- they differ on 94.06% of cells (mean |delta| 19.5\n"
                "ticks), so the model would be scored on a time coordinate it never saw.\n"
                "Pass --cell-t grid_center, or read `cell_t` out of the run's config.py.")
        pcfg = PatchConfig(cell_t=a.cell_t)
    print(f"tokenizer: cell_t={pcfg.cell_t} pw={pcfg.pw} pt={pcfg.pt} (from {src})", flush=True)

    n_ev = len(ident) if not a.max_events else min(a.max_events, len(ident))
    print(f"{n_ev} probe events from {os.path.basename(truth_path)}", flush=True)

    model_t, meta_t = load_probe_model(a.checkpoint, weights=a.weights)
    model_r, _ = load_probe_model(a.checkpoint, random_init=True,
                                  random_seed=a.random_seed)
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
        ce = read_coeff_event(shard, _position_of(shard, ev))
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

    # `corpus` is recorded because it was not, and an A/B whose entire independent
    # variable IS the corpus produced rows that never said which one they read —
    # it had to be recovered from `n_patch`. `weights` is the weight FILE
    # BASENAME, which for an export dir is always "model.bin" whatever it holds,
    # so it cannot distinguish EMA from raw on its own; `requested_weights` and
    # `weights_warning` record the case where the two disagree. `seeds` is a
    # SYSTEMATIC lever, not a nuisance parameter — `oof` is averaged over seeds,
    # worth about +0.05 on the trained arm going 1 -> 3 — so rows with different
    # `seeds` must never be compared, and `stale` says whether the truth artifact
    # still matched the corpus it was used with.
    base = dict(tag=a.tag, checkpoint=os.path.abspath(a.checkpoint),
                corpus=os.path.abspath(corpus),
                layer=a.layer, weights=meta_t["weights"],
                requested_weights=meta_t.get("requested_weights"),
                # What the export SAYS it holds, from the source checkpoint it
                # recorded: True/False/None-for-unknown. `requested_weights` is
                # only what was asked for; this is what was got, and a row with
                # weights_are_ema=None is unattributed rather than raw.
                weights_are_ema=meta_t.get("weights_are_ema"),
                weights_source=meta_t.get("weights_source"),
                weights_warning=meta_t.get("warning"),
                random_seed=a.random_seed,
                cell_t=pcfg.cell_t, pw=pcfg.pw, pt=pcfg.pt,
                qtot_min=float(cfg.get("qtot_min", -1)),
                dom_threshold=float(cfg.get("dom_threshold", -1)),
                folds=a.folds, epochs=a.epochs, seeds=a.seeds,
                n_patch=int(len(y)), n_events=int(len(np.unique(event))),
                truth=os.path.abspath(truth_path),
                stale=list(stale), when=int(time.time()),
                # --- fields that change the number and were NOT being recorded ---
                # `--dataset-name` selects which shard files are opened (see the
                # glob below), so two rows with different values are measuring
                # different data. It was the clearest violation of this module's
                # own docstring: "Everything that changes a number is echoed into
                # every results row."
                dataset_name=a.dataset_name,
                # The two digests that actually pin the split. `stale` reports
                # whether they MATCHED at load time; it does not say what they
                # were, so a row could not be compared against another row's split
                # after the fact.
                holdout_sha256=str(cfg.get("holdout_json_sha256", "")),
                corpus_ident_sha256=str(cfg.get("corpus_ident_sha256", "")),
                # Which weights, by content rather than by filename.
                weights_digest=_weights_digest_or_none(model_t),
                # Which code. fisher_r's definition lives in helix/probe/, so two
                # rows from different commits are not necessarily the same metric.
                code=_provenance_or_none())

    seeds = tuple(range(a.seeds))
    out_rows = []
    wanted = [p.strip() for p in a.probes.split(",") if p.strip()]

    if "mlp" in wanted:
        row = dict(base, probe="mlp")
        # ONE design resident at a time. mlp_designs(geo, arms) built all four up
        # front — 4 x 2064 dims x 2.5M rows x 4 B is ~62 GB of concatenated
        # copies on top of the 60 GB the three raw arms already hold, which
        # OOM-killed a 200 GB node before triangulate could start.
        for name in ("geo", "trained", "random", "raw"):
            if name != "geo" and arms.get(name) is None:
                continue
            X = mlp_designs(geo, {} if name == "geo" else {name: arms[name]})[name]
            oof, info = fit_probe(X, y, event, plane, n_folds=a.folds,
                                  epochs=a.epochs, seeds=seeds)
            r, rs, minfo = fisher_r(y, oof, event, plane)
            row[name] = dict(fisher_r=round(r, 4),
                             per_group_r_std=round(float(rs.std()), 4),
                             n_groups=minfo["n_groups"],
                             stop_epoch=round(info["mean_stop_epoch"], 1))
            print(f"  [mlp] {name:8s} fisher_r={r:+.4f}", flush=True)
            del X, oof
        if "geo" in row and "trained" in row:
            row["d_over_geo_r"] = round(row["trained"]["fisher_r"] - row["geo"]["fisher_r"], 4)
        _emit(a.out, row)
        out_rows.append(row)

    if "triangulate" in wanted:
        import gc
        for k in ("random", "raw"):          # only `trained` is used from here
            arms.pop(k, None)
        gc.collect()
        d = triangulate_designs(geo, arms["trained"], plane, cat("tick"),
                                cat("wire"), event)
        row = dict(base, probe="triangulate")
        for name in list(d):
            X = d.pop(name)                  # hand off; do not keep a second ref
            oof, info = fit_probe(X, y, event, plane, n_folds=a.folds,
                                  epochs=a.epochs, seeds=seeds)
            r, rs, minfo = fisher_r(y, oof, event, plane)
            row[name] = dict(fisher_r=round(r, 4),
                             per_group_r_std=round(float(rs.std()), 4),
                             n_groups=minfo["n_groups"],
                             stop_epoch=round(info["mean_stop_epoch"], 1))
            print(f"  [tri] {name:8s} fisher_r={r:+.4f}", flush=True)
            del X, oof
            gc.collect()
        _emit(a.out, row)
        out_rows.append(row)

    print(f"wrote {len(out_rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
