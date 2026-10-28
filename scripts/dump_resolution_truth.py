#!/usr/bin/env python3
"""Stage 0 of the resolution evaluation: fixed-grid truth for a held-out split.

    python scripts/dump_resolution_truth.py --out <dir> [--corpus C] [--source S]
        [--split probe] [--workers 16]

For each event in ``<corpus>/holdout.json[split]``, reads the simulation's
``hits`` (per-pixel truth charge and its Geant4 group -> track) and ``step``
(deposited energy per track), plus the event's kept coefficients, and writes
``<out>/evNNN.npz`` (see :func:`helix.probe.resolution.event_truth`). Depends on
no tokenizer and no checkpoint, so every model is scored on the same cells.
Events already written are skipped, so an interrupted run resumes.

Defaults: ``--corpus`` from HELIX_CORPUS; ``--source`` is the directory holding
``hits/`` and ``step/``, by default the parent of HELIX_SENSOR_ROOT.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _position_of(shard, event_id):
    import h5py
    with h5py.File(shard, "r") as f:
        ids = f["ident"]["event"][:]
    pos = int(np.searchsorted(ids, event_id))
    if pos >= len(ids) or int(ids[pos]) != int(event_id):
        raise KeyError(f"{shard}: no event {event_id}")
    return pos


def _one(job):
    i, ent, a = job
    out = os.path.join(a["out"], f"ev{i:03d}.npz")
    if os.path.exists(out):
        return i, "exists"
    import h5py
    import hdf5plugin  # noqa: F401  (step/hits shards use a plugin filter)
    from helix.core.coeff_io import read_coeff_event
    from helix.model.tokenize import PatchConfig, tick_of_tau
    from helix.probe.resolution import FT, FW, event_truth, fkey
    from helix.probe.truth import PLANES, decode_hits_plane

    tag = ent["source_file"].replace("sim_wire_sensor_", "").replace(".h5", "")
    shard = os.path.join(a["corpus"], f"sim_wire_coeff_{tag}.h5")
    ce = read_coeff_event(shard, _position_of(shard, ent["event"]))
    hp = os.path.join(a["source"], "hits", ent["run"], f"sim_wire_hits_{tag}.h5")
    sp = os.path.join(a["source"], "step", ent["run"], f"sim_wire_step_{tag}.h5")
    key = f"event_{ce.event:03d}"
    G, W, T, Q, TRK, E = [], [], [], [], [], {}
    with h5py.File(sp, "r") as fs, h5py.File(hp, "r") as fh:
        for vk in sorted(x for x in fh[key] if x.startswith("volume_")):
            vi = int(vk.split("_")[1])
            de = fs[key][vk]["de"][:].astype(np.float64)
            d2g = fh[key][vk]["deposit_to_group"][:].astype(np.int64)
            g2t = fh[key][vk]["group_to_track"][:].astype(np.int64)
            tu, ti = np.unique(g2t[d2g], return_inverse=True)
            for t_, e_ in zip(tu.tolist(), np.bincount(ti, weights=de).tolist()):
                E[vi * 10_000_000 + t_] = e_                    # one id per particle, global
            for pi, pl in enumerate(PLANES):
                if pl not in fh[key][vk]:
                    continue
                s = decode_hits_plane(fh[key][vk][pl])
                G.append(np.full(len(s["q"]), vi * 3 + pi)); W.append(s["wire"]); T.append(s["tick"])
                Q.append(s["q"]); TRK.append(vi * 10_000_000 + g2t[s["group"]])
    cfg = PatchConfig(cell_t="grid_center")
    k = ce.band < cfg.n_bands
    ct = tick_of_tau(ce.tau[k], ce.plane_gid[k], ce.band[k], cfg)
    cells = fkey(ce.plane_gid[k], ce.wire[k] // FW, np.floor(ct / FT).astype(np.int64))
    pix = dict(g=np.concatenate(G), w=np.concatenate(W), t=np.concatenate(T), q=np.concatenate(Q),
               trk=np.concatenate(TRK), E=E)
    tr = event_truth(pix, cells, np.random.default_rng(a["seed"] + i))
    np.savez_compressed(out, mrows=tr["mrows"], mq=tr["mq"], wkey=tr["wkey"], wq=tr["wq"], wid=tr["wid"],
                        meta=json.dumps(tr["meta"]), shard=shard, event=ent["event"])
    kinds = [m["kind"] for m in tr["meta"]]
    return i, f"rows {len(tr['mrows'])}  iso {kinds.count('iso')} pair {kinds.count('pair')} bg {kinds.count('bg')}"


def main():
    from helix.paths import root
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default=None, help="default: HELIX_CORPUS")
    ap.add_argument("--source", default=None, help="dir with hits/ and step/; default: parent of HELIX_SENSOR_ROOT")
    ap.add_argument("--split", default="probe")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    corpus = a.corpus or str(root("HELIX_CORPUS"))
    source = a.source or str(root("HELIX_SENSOR_ROOT").parent)
    os.makedirs(a.out, exist_ok=True)
    events = json.load(open(os.path.join(corpus, "holdout.json")))[a.split]
    args = dict(out=a.out, corpus=corpus, source=source, seed=a.seed)
    jobs = [(i, e, args) for i, e in enumerate(events)]
    with Pool(a.workers) as pool:
        for i, msg in pool.imap_unordered(_one_safe, jobs):
            print(f"{i:4d} {msg}", flush=True)


def _one_safe(job):
    try:
        return _one(job)
    except Exception as e:                       # an event without sim files is skipped, and said
        return job[0], f"SKIP {type(e).__name__}: {str(e)[:120]}"


if __name__ == "__main__":
    main()
