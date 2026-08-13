"""Stage 1: derive per-pixel probe truth for a held-out split.

Reads the simulation's ``hits`` + ``step`` shards for the events named in a
corpus's ``holdout.json`` and writes one self-describing artifact beside the
corpus. Depends on the corpus only for WHICH events to dump — never for their
content — which is why the artifact survives a corpus rebuild.

    python scripts/dump_probe_truth.py \
        --corpus /sdf/data/neutrino/omara/coeff_tpc/run_0027575715 \
        --source /sdf/data/neutrino/doraemon/wire_test_00_00_02 \
        --split probe

Writes ``<corpus>/truth/probe_truth_<split>.h5``.

Three things this refuses to do quietly:

* **Locate events by position.** The dump drives off the manifest's
  ``(source_file, event)`` triples and opens exactly that file and group.
  ``idx // 200`` is wrong for this corpus — ``sim_wire_sensor_0065.h5`` holds 199
  events because ``event_167`` is absent.
* **Ship an unverifiable artifact.** Three digests tie it to the split, the
  manifest and the source shards, and the along-wire axis is frozen into
  ``/config`` after being fitted over the whole split.
* **Ship a bad join.** Charge-weighted coverage of each event's pixels by its
  corpus event's cells is measured per event; below ``--min-coverage`` on band 0
  the dump fails. Measured on a matched event: 0.999. On a deliberately
  mismatched one: 0.372.
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


def _sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _source_manifest_sha256(paths):
    """Digest of (name, size, mtime) for every source shard read."""
    rows = []
    for p in sorted(paths):
        st = os.stat(p)
        rows.append(f"{os.path.basename(p)}|{st.st_size}|{st.st_mtime_ns}")
    return _sha256_text("\n".join(rows))


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



_PIX_KEYS = ("gid", "wire", "tick", "qtot", "ftop", "b1")
_FIT_KEYS = ("gid", "y", "z", "wire")


def _ckpt_save(path, rows):
    """Checkpoint completed events so a preemption costs minutes, not the run.

    Written atomically: a run killed mid-write leaves the previous checkpoint
    intact rather than a truncated one. Measured need — a 388-event dump was
    preempted at event 175 and lost everything.
    """
    import os
    blob = {"n": np.array([len(rows)])}
    for i, r in enumerate(rows):
        for k in _PIX_KEYS:
            blob[f"{i}.{k}"] = r[k]
        for k in _FIT_KEYS:
            blob[f"{i}._fit.{k}"] = r["_fit"][k]
        blob[f"{i}.ident"] = np.array([r["ident"][0], r["ident"][1], str(r["ident"][2])])
        blob[f"{i}.cov"] = np.asarray(r["_cov"], np.float64)
    np.savez(path + ".tmp", **blob)
    os.replace(path + ".tmp", path)


def _ckpt_load(path):
    d = np.load(path, allow_pickle=False)
    out = []
    for i in range(int(d["n"][0])):
        r = {k: d[f"{i}.{k}"] for k in _PIX_KEYS}
        r["_fit"] = {k: d[f"{i}._fit.{k}"] for k in _FIT_KEYS}
        run, src, ev = d[f"{i}.ident"]
        r["ident"] = (str(run), str(src), int(ev))
        r["_cov"] = d[f"{i}.cov"].tolist()
        out.append(r)
    return out


def _dequantise(pos, vol_range):
    lo, hi = vol_range[:, 0], vol_range[:, 1]
    return lo + pos.astype(np.float64) / 65535.0 * (hi - lo)


def dump(args):
    import h5py
    try:
        import hdf5plugin        # noqa: F401  (registers the source shards' codec)
    except ImportError:
        raise SystemExit(
            "hdf5plugin is required to read the doraemon hits/step shards "
            "(pip install 'helix-tpc[probe]'); without it h5py raises "
            "\"can't open directory .../hdf5/lib/plugin\" on the first read.")

    from helix.core.coeff_io import read_coeff_event
    from helix.model.tokenize import PatchConfig, cell_key, pixel_cells
    from helix.probe.alongwire import fit_alongwire
    from helix.probe.truth import PLANES, decode_hits_plane, group_centroids, pixel_truth

    corpus = os.path.abspath(args.corpus)
    hj = os.path.join(corpus, "holdout.json")
    holdout_bytes = open(hj, "rb").read()
    manifest = json.loads(holdout_bytes)
    if args.split not in manifest:
        raise SystemExit(
            f"{hj} has no '{args.split}' list (has {sorted(k for k in manifest if isinstance(manifest[k], list))}). "
            f"train is stored as the complement and is not dumped.")
    events = manifest[args.split]
    if args.limit:
        events = events[:args.limit]
    print(f"{len(events)} events in split '{args.split}'", flush=True)

    cfg = PatchConfig(cell_t=args.cell_t)
    ev_rows, touched = [], set()
    cov_report = []

    # Resume point. The manifest order is fixed, so the number of completed
    # events is a sufficient cursor. The name carries every parameter that
    # changes CONTENT, so a checkpoint from a different qtot_min or cell_t is
    # never silently reused.
    ck = os.path.join(args.work_dir or os.path.join(corpus, "truth"),
                      f"_dump_ckpt_{args.split}_q{args.qtot_min:g}"
                      f"_d{args.dom_threshold:g}_{args.cell_t}.npz")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    start = 0
    if os.path.exists(ck) and not args.restart:
        ev_rows = _ckpt_load(ck)
        cov_report = [r["_cov"] for r in ev_rows]
        start = len(ev_rows)
        print(f"resuming after {start} events (from {os.path.basename(ck)})", flush=True)

    for n, ident in enumerate(events):
        if n < start:
            continue
        run, src, ev = ident["run"], ident["source_file"], int(ident["event"])
        tag = src.replace("sim_wire_sensor_", "").replace(".h5", "")
        hp = os.path.join(args.source, "hits", run, f"sim_wire_hits_{tag}.h5")
        sp = os.path.join(args.source, "step", run, f"sim_wire_step_{tag}.h5")
        touched.update((hp, sp))

        rows = {k: [] for k in ("gid", "wire", "tick", "qtot", "ftop", "b1")}
        # Deposit-level samples for the along-wire fit, kept SEPARATE from the
        # pixel rows. The reference fits wire ~ a*b1_y + b*b1_z, but b1 is a
        # GROUP centroid averaged over a group that spans many wires, so it is a
        # proxy for the deposit's position. Measured: fitting on b1 gives
        # residuals of 15-90 wires and axes off by ~0.01, while fitting on the
        # deposits' own positions gives 0.30 wires and exactly +-60/0 degrees.
        # The axis defines the target, so the biased fit biases u.
        fit_rows = {k: [] for k in ("gid", "y", "z", "wire")}
        with h5py.File(sp, "r") as fs, h5py.File(hp, "r") as fh:
            key = f"event_{ev:03d}"
            if key not in fs or key not in fh:
                raise SystemExit(f"{run}/{src} event {ev}: {key} absent from "
                                 f"{'step' if key not in fs else 'hits'} — the "
                                 f"manifest names an event the source lacks")
            vr = fs["config"]["volume_ranges"][:]
            gs, gh = fs[key], fh[key]
            for vk in sorted(k for k in gs if k.startswith("volume_")):
                vi = int(vk.split("_")[1])
                xyz = _dequantise(gs[vk]["positions"][:], vr[vi])
                chg = gs[vk]["charge"][:].astype(np.float64)
                d2g = gh[vk]["deposit_to_group"][:]
                cen = group_centroids(xyz, chg, d2g)
                for pi, pl in enumerate(PLANES):
                    if pl not in gh[vk]:
                        continue
                    # deposit -> its group's row in THIS plane -> that row's wire
                    grp_ids = gh[vk][pl]["group_ids"][:]
                    cwire = gh[vk][pl]["center_wires"][:].astype(np.float64)
                    order = np.argsort(grp_ids)
                    sg = grp_ids[order]
                    j = np.searchsorted(sg, d2g)
                    ok = j < len(sg)
                    j = np.clip(j, 0, max(len(sg) - 1, 0))
                    ok &= (sg[j] == d2g)
                    if ok.sum():
                        fit_rows["gid"].append(np.full(int(ok.sum()), vi * 3 + pi, np.int64))
                        fit_rows["y"].append(xyz[ok, 1])
                        fit_rows["z"].append(xyz[ok, 2])
                        fit_rows["wire"].append(cwire[order[j[ok]]])
                    s = decode_hits_plane(gh[vk][pl])
                    w, t, q, f, b1 = pixel_truth(s, cen, args.qtot_min)
                    if not len(w):
                        continue
                    rows["gid"].append(np.full(len(w), vi * 3 + pi, np.int8))
                    for k, v in (("wire", w), ("tick", t), ("qtot", q),
                                 ("ftop", f), ("b1", b1)):
                        rows[k].append(v)
        if not rows["gid"]:
            raise SystemExit(f"{run}/{src} event {ev}: no pixels above "
                             f"qtot_min={args.qtot_min}")
        R = {k: np.concatenate(v) for k, v in rows.items()}

        # --- the join check, per event -------------------------------------
        cshard = os.path.join(corpus, f"sim_wire_coeff_{tag}.h5")
        with h5py.File(cshard, "r") as f:
            bl = f["config"]["band_lengths"][:]
        ce = read_coeff_event(cshard, _position_of(cshard, ev))
        m = ce.band < cfg.n_bands
        have = np.unique(cell_key(ce.plane_gid[m], ce.band[m], ce.wire[m], ce.tau[m], cfg))
        pc = pixel_cells(R["gid"], R["wire"], R["tick"], bl, cfg)
        cov = []
        for b in range(cfg.n_bands):
            ins = np.isin(pc[:, b], have)
            cov.append(float((R["qtot"] * ins).sum() / max(R["qtot"].sum(), 1e-9)))
        cov_report.append(cov)
        if cov[0] < args.min_coverage:
            raise SystemExit(
                f"{run}/{src} event {ev}: band-0 charge-weighted coverage "
                f"{cov[0]:.3f} < {args.min_coverage} — the pixel->cell join is "
                f"wrong (a mismatched event scores ~0.37, a good one ~0.999). "
                f"Check toff/delta/lev, band_lengths, or the /ident pairing.")

        R["ident"] = (run, src, ev)
        R["_cov"] = cov
        R["_fit"] = {k: (np.concatenate(v) if v else np.empty(0))
                     for k, v in fit_rows.items()}
        ev_rows.append(R)
        if (n + 1) % 25 == 0 or n + 1 == len(events):
            _ckpt_save(ck, ev_rows)
            print(f"  {n+1}/{len(events)} events, "
                  f"band0 cov {np.mean([c[0] for c in cov_report]):.4f}"
                  f"  [checkpointed]", flush=True)

    # --- one along-wire fit over the whole split ---------------------------
    F = {k: np.concatenate([r["_fit"][k] for r in ev_rows]) for k in ("gid", "y", "z", "wire")}
    F["event"] = np.concatenate([np.full(len(r["_fit"]["gid"]), i)
                                 for i, r in enumerate(ev_rows)])
    vecs, diag = fit_alongwire(F["gid"], F["y"], F["z"], F["wire"], event=F["event"])
    aw = np.zeros((6, 2), np.float64)
    aw_n = np.zeros(6, np.int64)
    aw_res = np.zeros(6, np.float32)
    for g, v in vecs.items():
        aw[g] = v
        aw_n[g] = diag[g]["n"]
        aw_res[g] = diag[g]["resid_rms_wires"]
    print("along-wire (frozen):")
    for g in range(6):
        print(f"  gid {g}: ({aw[g,0]:+.4f}, {aw[g,1]:+.4f})  n={aw_n[g]:,}  "
              f"resid {aw_res[g]:.2f} wires")

    # --- write --------------------------------------------------------------
    outdir = os.path.join(corpus, "truth")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"probe_truth_{args.split}.h5")
    ident_txt = "\n".join(sorted(f"{r['ident'][0]}/{r['ident'][1]}/{r['ident'][2]}"
                                 for r in ev_rows))
    offs = np.cumsum([0] + [len(r["gid"]) for r in ev_rows]).astype(np.int64)

    import helix
    with h5py.File(out, "w") as f:
        c = f.create_group("config")
        c.attrs["truth_schema_version"] = 1
        c.attrs["qtot_min"] = float(args.qtot_min)
        c.attrs["dom_threshold"] = float(args.dom_threshold)
        c.attrs["cell_t_checked"] = args.cell_t
        c.attrs["tokenizer_independent"] = True     # per-pixel; see helix/probe/truth.py
        c.attrs["corpus_dir"] = corpus
        c.attrs["source_root"] = os.path.abspath(args.source)
        c.attrs["split_role"] = args.split
        c.attrs["corpus_ident_sha256"] = _sha256_text(ident_txt)
        c.attrs["holdout_json_sha256"] = hashlib.sha256(holdout_bytes).hexdigest()
        c.attrs["source_manifest_sha256"] = _source_manifest_sha256(touched)
        c.attrs["provenance_json"] = json.dumps(dict(
            builder="dump_probe_truth.py",
            helix_version=getattr(helix, "__version__", "unknown"),
            built_at=int(time.time()),
            band0_coverage_mean=float(np.mean([c0[0] for c0 in cov_report])),
            band0_coverage_min=float(np.min([c0[0] for c0 in cov_report])),
        ), sort_keys=True)
        c.create_dataset("alongwire", data=aw.astype(np.float32))
        c.create_dataset("alongwire_n", data=aw_n)
        c.create_dataset("alongwire_resid_rms", data=aw_res)
        c.create_dataset("coverage", data=np.asarray(cov_report, np.float32))

        p = f.create_group("pix")
        for k, dt in (("gid", np.int8), ("wire", np.int32), ("tick", np.int32),
                      ("qtot", np.float32), ("ftop", np.float32)):
            p.create_dataset(k, data=np.concatenate([r[k] for r in ev_rows]).astype(dt),
                             compression="gzip", compression_opts=1)
        p.create_dataset("b1", data=np.concatenate([r["b1"] for r in ev_rows]).astype(np.float32),
                         compression="gzip", compression_opts=1)
        p.create_dataset("event_offset", data=offs)

        i = f.create_group("ident")
        st = h5py.string_dtype()
        i.create_dataset("run", data=np.array([r["ident"][0] for r in ev_rows], dtype=st))
        i.create_dataset("source_file", data=np.array([r["ident"][1] for r in ev_rows], dtype=st))
        i.create_dataset("event", data=np.array([r["ident"][2] for r in ev_rows], np.int64))

    print(f"wrote {out}  ({offs[-1]:,} pixels, {len(ev_rows)} events, "
          f"{os.path.getsize(out)/2**20:.0f} MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--source", required=True,
                    help="doraemon root holding hits/ and step/")
    ap.add_argument("--split", default="probe")
    ap.add_argument("--qtot-min", type=float, default=250.0,
                    help="TARGET parameter: pixels below this do not define y")
    ap.add_argument("--dom-threshold", type=float, default=0.5,
                    help="TARGET parameter: f_top at or above this is 'dominant'")
    ap.add_argument("--cell-t", default="grid_center",
                    choices=("grid_center", "centroid"),
                    help="only affects the coverage CHECK; cell identity is "
                         "mode-independent, which is why this artifact is not")
    ap.add_argument("--min-coverage", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0, help="first N events only (smoke)")
    ap.add_argument("--work-dir", default=None, help="where the resume checkpoint lives")
    ap.add_argument("--restart", action="store_true", help="ignore any checkpoint")
    a = ap.parse_args(argv)
    if a.limit:
        print(f"--limit {a.limit}: dumping a SUBSET, not a usable artifact")
    dump(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
