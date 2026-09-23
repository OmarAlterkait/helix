"""Stage 2: probe a checkpoint against the per-pixel truth artifact.

    python scripts/run_probe.py --checkpoint /path/fm.pt \
        --corpus <corpus run dir> \
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
    """The scored weights' identity. One implementation, in helix.model.artifact.

    It used to live here, which is how the export script came to grow a second,
    incompatible one -- a script is not a place a shared definition can be found
    from.
    """
    from helix.model.artifact import weights_digest
    return weights_digest(model.state_dict())


#: The feature arms. Cached at FULL precision by default, and that default was
#: earned: at float16 a resumed run reproduced `trained`, `geo` and `random` to
#: the printed 4 decimals but moved `raw` from +0.0044 to +0.0045. The shift is
#: ~1e-4, which is 20x below the probe's own seed-to-seed sigma (0.0021) and
#: changes no conclusion -- but it makes `cache_resumed_events` a field that
#: moves the number, and this file's whole contract is that everything which
#: moves a number is recorded and nothing else does. A resumed run must be the
#: same run.
#:
#: `--cache-half` is there for when disk is the binding constraint. It halves a
#: cache that is otherwise ~n_patch x feat_dim x 3 arms x 4 B (about 62 GB for
#: the 388-event split at feat_dim 2048), at the cost above.
_CACHE_ARMS = ("Xtr", "Xrn", "Xraw")


def _cache_key(**parts):
    """A directory name that changes whenever the cached features would.

    Everything that feeds ``features_at_layer`` is in here, INCLUDING the
    random-init control's own weight digest: the `random` arm lives in the same
    cache, and reusing one seed's null under another seed would silently compare
    two arms against two different floors. What is deliberately NOT in here is
    ``--max-events``: events are cached by index in the truth artifact's own
    order, so a 400-event run reuses a 100-event run's chunks.

    ``serial``/``rope_split``/``gp``/``gd`` are in here for a reason the weight
    digest cannot cover: CLAUDE.md records that those four "leave NO trace in
    the weights", so two models that differ only in attention block size have
    the SAME digest and produced the same key. They do not produce the same
    features -- the grouped attention's partition is a function of ``gp``/``gd``
    -- so a block-size sweep run into one ``--cache-dir`` silently scored every
    arm against the first arm's features.
    """
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_chunks(d):
    """``[(start, end, path)]`` in order, for the chunks present."""
    import glob as _glob
    out = []
    for f in sorted(_glob.glob(os.path.join(d, "ev_*.npz"))):
        lo, hi = os.path.basename(f)[3:-4].split("_")
        out.append((int(lo), int(hi), f))
    return out


def _cache_resume(d):
    """``(packs, next_event)`` from the CONTIGUOUS prefix of cached chunks.

    The saving that makes this worth having is the FITS (measured: geo 36 min,
    trained 58 min), not the extraction (12.5 min) -- see `arms.json`.

    Contiguous on purpose. A gap means a chunk was lost or a run died mid-write,
    and silently skipping the missing events would produce a result whose
    ``n_events`` looks right in the row while the features behind it are a
    different set. Trailing chunks past a gap are ignored, not deleted -- a later
    run with the same key picks them up once the gap is refilled.
    """
    packs, nxt = [], 0
    for lo, hi, f in _cache_chunks(d):
        if lo != nxt:
            print(f"  cache: gap at event {nxt} (next chunk starts {lo}); "
                  f"re-extracting from there", flush=True)
            break
        z = np.load(f)
        n = int(z["_n_packs"])
        for j in range(n):
            pre = f"p{j}_"
            pk = {}
            for k in z.files:
                if not k.startswith(pre):
                    continue
                col = k[len(pre):]
                v = z[k]
                if col in _CACHE_ARMS and v.dtype != np.float32:
                    # Widen a half cache; a full one is already float32 and an
                    # unconditional astype would copy all 44 GB for nothing.
                    v = v.astype(np.float32)
                pk[col] = v
            pk["n_bands"] = int(pk["n_bands"])
            packs.append(pk)
        nxt = hi
    return packs, nxt


def _cache_write(d, lo, hi, chunk, half=False):
    """Write one chunk atomically. A half-written .npz is worse than none."""
    os.makedirs(d, exist_ok=True)
    flat = {"_n_packs": np.int64(len(chunk))}
    for j, pk in enumerate(chunk):
        for k, v in pk.items():
            v = np.asarray(v)
            flat[f"p{j}_{k}"] = v.astype(np.float16) if (half and k in _CACHE_ARMS) else v
    tmp = os.path.join(d, f".ev_{lo:05d}_{hi:05d}.tmp.npz")
    np.savez(tmp, **flat)
    os.replace(tmp, os.path.join(d, f"ev_{lo:05d}_{hi:05d}.npz"))


def _arm_key(probe, name, a):
    """Identity of ONE fitted arm.

    The fit parameters belong here, not in the feature cache's directory name:
    `folds`/`epochs`/`seeds` change the NUMBER but not the features, so changing
    them must invalidate a fitted arm while still reusing the extraction that
    cost the GPU hour.
    """
    return f"{probe}:{name}:f{a.folds}:e{a.epochs}:s{a.seeds}:r{a.random_seed}"


def _arms_load(cache_dir):
    """Arms already fitted under this feature cache, or an empty map."""
    if not cache_dir:
        return {}
    p = os.path.join(cache_dir, "arms.json")
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}


def _arms_save(cache_dir, fitted):
    """Persist after EVERY arm. The point is that the next one may not finish.

    Extraction resume was only half the problem: the four mlp arms are ~1h20m
    each at 2064 dims over 2.5M rows, `_emit` writes a row only once all four
    are done, and a job that walls on its time limit three arms in loses all
    three. That is the same failure `_emit`'s own docstring was written about
    ("an OOM in triangulate discarded a complete four-arm mlp result"), one
    level further in.
    """
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    tmp = os.path.join(cache_dir, ".arms.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(fitted, fh, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(cache_dir, "arms.json"))


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
    ap.add_argument("--cache-dir", default=None,
                    help="cache extracted features here and RESUME from them. The "
                         "extraction loop is the long pole — 2.5M patches over 388 "
                         "events — and a preemption 40 minutes in used to discard "
                         "all of it. Off by default because the cache is large "
                         "(~n_patch x feat_dim x 3 arms x 4 B).")
    ap.add_argument("--cache-half", action="store_true",
                    help="halve the cache by storing the feature arms at "
                         "float16. Measured cost: a resumed run moved the `raw` "
                         "arm by 1e-4 (+0.0044 -> +0.0045), which is 20x under "
                         "the seed-to-seed sigma but is NOT the same number. Use "
                         "only when disk is the binding constraint.")
    ap.add_argument("--cache-every", type=int, default=25,
                    help="flush a cache chunk every N events; the resume floor")
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

    # A model trained on one basis scored against another is not an error
    # anywhere else in the stack: the shards load, the tokenizer runs, and a
    # plausible number comes out. `coeff_tpc` (7f954a84…) and `coeff_tpc_r1`
    # (8c4542b6…) differ only by the occupancy gate. So check it here, where the
    # two are finally in the same process.
    _prov = meta_t.get("provenance") or {}
    if _prov.get("corpus"):
        from helix.data.identity import check_corpus_matches, corpus_identity
        print(check_corpus_matches(_prov["corpus"], corpus_identity(
            corpus, dataset_name=a.dataset_name), where=a.checkpoint), flush=True)

    cache_dir, start = None, 0
    packs = []
    if a.cache_dir:
        cache_dir = os.path.join(a.cache_dir, _cache_key(
            trained=_weights_digest_or_none(model_t),
            random=_weights_digest_or_none(model_r),
            layer=a.layer, cell_t=pcfg.cell_t, pw=pcfg.pw, pt=pcfg.pt,
            # Not recoverable from the weights -- see the docstring.
            serial=getattr(model_t, "gp", None) is not None,
            rope_split=getattr(model_t, "rope_split", None),
            gp=getattr(model_t, "gp", None), gd=getattr(model_t, "gd", None),
            n_bands=pcfg.n_bands, corpus=corpus, dataset_name=a.dataset_name,
            # Precision is part of the key: a half cache and a full one hold
            # different numbers, so they must not be read as one another's.
            half=bool(a.cache_half),
            truth=os.path.abspath(truth_path),
            corpus_ident=str(cfg.get("corpus_ident_sha256", "")),
            dom_threshold=float(cfg.get("dom_threshold", 0.5))))
        packs, start = _cache_resume(cache_dir)
        start = min(start, n_ev)
        print(f"cache {cache_dir}: {len(packs)} packs, resuming at event {start}",
              flush=True)
        # A cached pack's `event` column is its index in the truth artifact's
        # order, which is what `fit_probe` groups folds by -- so a resumed run
        # and an uninterrupted one produce the SAME folds, not merely the same
        # number of them.
        packs = [p for p in packs if int(p["event"][0]) < n_ev]

    pending, pend_lo = [], start
    for i in range(start, n_ev):
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
        pending.append(pr)
        del ft, fr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cache_dir and (len(pending) >= a.cache_every or i + 1 == n_ev):
            _cache_write(cache_dir, pend_lo, i + 1, pending, half=a.cache_half)
            pending, pend_lo = [], i + 1
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
    # it had to be recovered from `n_patch`. `weights` is now what was USED
    # ("ema" / "raw" / "unknown"), not the weight file's basename — that was
    # always "model.bin" for an export dir whatever it held, so it could not
    # distinguish EMA from raw at all; `requested_weights` and `weights_warning`
    # record the case where asked-for and got disagree, and "unknown" means the
    # checkpoint could not say (promote it with scripts/export_artifact.py). `seeds` is a
    # SYSTEMATIC lever, not a nuisance parameter — `oof` is averaged over seeds,
    # worth about +0.05 on the trained arm going 1 -> 3 — so rows with different
    # `seeds` must never be compared, and `stale` says whether the truth artifact
    # still matched the corpus it was used with.
    _wd = _weights_digest_or_none(model_t)
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
                # What the checkpoint says about ITSELF, when it is a helix eval
                # artifact. `weights_sha256` identifies the tensors with no
                # cooperation from the writer, so two rows claiming the same
                # checkpoint can be shown to have probed the same weights;
                # `ckpt_basis_digest` is the corpus the weights were TRAINED on,
                # checked against the one being read just below.
                ckpt_basis_digest=str((_prov.get("corpus") or {}).get("basis_digest", "")),
                ckpt_helix=str((_prov.get("helix") or {}).get("commit", "")),
                ckpt_helix_dirty=(_prov.get("helix") or {}).get("dirty"),
                # The artifact's own record of the weights it holds. Same
                # function as `weights_digest` below, computed at promotion time
                # over the saved tensors rather than here over the loaded model,
                # so the two MUST agree -- and `weights_digest_matches` says
                # whether they did, which is a real check on the load path.
                ckpt_weights_digest=str(_prov.get("weights_digest", "")),
                random_seed=a.random_seed,
                # Which cache these features came from, and how many events were
                # reused rather than extracted. A resumed run and an
                # uninterrupted one must produce the same number; recording it is
                # what makes that checkable after the fact rather than asserted.
                cache_key=(os.path.basename(cache_dir) if cache_dir else None),
                cache_resumed_events=int(start),
                cache_half=bool(a.cache_half),
                # How many arms were reused rather than fitted. Alongside
                # cache_resumed_events this makes a resumed row auditable: both
                # are expected to change nothing, and a row records enough to
                # check that rather than asking anyone to trust it.
                cache_arms_reused=len(_arms_load(cache_dir)),
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
                weights_digest=_wd,
                weights_digest_matches=(
                    None if not (_wd and _prov.get("weights_digest"))
                    else _wd == _prov["weights_digest"]),
                # Which code. fisher_r's definition lives in helix/probe/, so two
                # rows from different commits are not necessarily the same metric.
                code=_provenance_or_none())

    seeds = tuple(range(a.seeds))
    out_rows = []
    wanted = [p.strip() for p in a.probes.split(",") if p.strip()]
    fitted = _arms_load(cache_dir)
    if fitted:
        print(f"cache: {len(fitted)} arm(s) already fitted, reusing", flush=True)

    if "mlp" in wanted:
        row = dict(base, probe="mlp")
        # ONE design resident at a time. mlp_designs(geo, arms) built all four up
        # front — 4 x 2064 dims x 2.5M rows x 4 B is ~62 GB of concatenated
        # copies on top of the 60 GB the three raw arms already hold, which
        # OOM-killed a 200 GB node before triangulate could start.
        for name in ("geo", "trained", "random", "raw"):
            # The cached check comes FIRST. Its features are deliberately not
            # loaded when it is already fitted, so testing availability first
            # would `continue` past a completed arm and drop it from the row.
            k = _arm_key("mlp", name, a)
            if k in fitted:
                row[name] = fitted[k]
                print(f"  [mlp] {name:8s} fisher_r={row[name]['fisher_r']:+.4f} "
                      f"(cached)", flush=True)
                continue
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
            fitted[k] = row[name]
            _arms_save(cache_dir, fitted)
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
        tri_names = ("solo", "cross", "xwire")
        row = dict(base, probe="triangulate")
        d = triangulate_designs(geo, arms["trained"], plane, cat("tick"),
                                cat("wire"), event)
        for name in list(d):
            X = d.pop(name)                  # hand off; do not keep a second ref
            k = _arm_key("triangulate", name, a)
            if k in fitted:
                row[name] = fitted[k]
                print(f"  [tri] {name:8s} fisher_r={row[name]['fisher_r']:+.4f} "
                      f"(cached)", flush=True)
                del X
                gc.collect()
                continue
            oof, info = fit_probe(X, y, event, plane, n_folds=a.folds,
                                  epochs=a.epochs, seeds=seeds)
            r, rs, minfo = fisher_r(y, oof, event, plane)
            row[name] = dict(fisher_r=round(r, 4),
                             per_group_r_std=round(float(rs.std()), 4),
                             n_groups=minfo["n_groups"],
                             stop_epoch=round(info["mean_stop_epoch"], 1))
            fitted[k] = row[name]
            _arms_save(cache_dir, fitted)
            print(f"  [tri] {name:8s} fisher_r={r:+.4f}", flush=True)
            del X, oof
            gc.collect()
        _emit(a.out, row)
        out_rows.append(row)

    print(f"wrote {len(out_rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
