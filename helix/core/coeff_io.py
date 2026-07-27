"""Reference codec for the coeff corpus — CoeffEvent ⇄ flat-columnar HDF5 shard.

This is the **schema of record** and the **acceptance gate**: helix defines the
on-disk layout here and pins it with a round-trip identity test. pimm-data's
production `CoeffShardWriter`/`CoeffTPCReader` implement the *same* documented
layout independently (standalone h5py, no helix import) — kept in sync by a
cross-repo golden.

Layout (one modality per file, flat columnar — see COEFF_CORPUS_DESIGN.md §3):

    /config   attrs: n_events, dataset_name, file_index, global_event_offset,
                     readout_type, wavelet, dwt_level, dwt_mode, n_ticks_raw, pad,
                     sigma_norm, basis_digest, removal_json, threshold_json
              datasets: band_lengths (n_bands,), gids (G,), n_wires (G,),
                        [norm_sigma (G, n_bands)]
    /coord    band uint8, plane_gid uint8, wire int32, tau int32   (all (M,))
              event_offset int64 (n_events+1,)   ← slice [off[i]:off[i+1]]
              sigma_threshold float32 (n_events, G, n_bands)
    /value    float32 (M,)   RAW noisy coeff
    /ident    run (n_events,), source_file (n_events,), event int64 (n_events,)

Compression defaults to gzip (always readable); the layout — not the filter — is
the contract, so pimm-data may write blosc-zstd over the same schema.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import h5py

from helix.core.coeff_event import CoeffEvent
from helix.core.provenance import BasisDescriptor

_STR = h5py.string_dtype(encoding="utf-8")


def _ds(grp, name, data, comp):
    kw = dict(compression=comp, compression_opts=4) if comp == "gzip" else {}
    grp.create_dataset(name, data=data, **kw)


def _basis_from_config(cfg) -> BasisDescriptor:
    b = BasisDescriptor(
        wavelet=str(cfg.attrs["wavelet"]),
        level=int(cfg.attrs["dwt_level"]),
        mode=str(cfg.attrs["dwt_mode"]),
        n_ticks_raw=int(cfg.attrs["n_ticks_raw"]),
        pad=int(cfg.attrs["pad"]),
        band_lengths=tuple(int(x) for x in cfg["band_lengths"][:]),
        removal=json.loads(cfg.attrs.get("removal_json", "{}")),
        threshold=json.loads(cfg.attrs.get("threshold_json", "{}")),
        sigma_norm=float(cfg.attrs["sigma_norm"]),
    )
    b.validate()                         # fail loudly if band_lengths drifted from the basis
    stored = cfg.attrs.get("basis_digest")
    if stored and str(stored) != b.digest():
        raise ValueError(
            f"basis_digest mismatch: /config says {stored!r} but the basis attrs "
            f"hash to {b.digest()!r} — the shard's declared identity disagrees with its basis.")
    return b


# ---- pure per-event (de)serialization — the read contract -----------------

def coeff_event_to_arrays(ce: CoeffEvent) -> dict:
    """One CoeffEvent → a flat dict of arrays+scalars (what the reader emits)."""
    return dict(
        band=ce.band, plane_gid=ce.plane_gid, wire=ce.wire, tau=ce.tau, value=ce.value,
        gids=ce.gids, n_wires=ce.n_wires, sigma_threshold=ce.sigma_threshold,
        run=ce.run, source_file=ce.source_file, event=ce.event,
    )


def arrays_to_coeff_event(d: dict, basis: BasisDescriptor) -> CoeffEvent:
    """Flat dict + basis → CoeffEvent (inverse of `coeff_event_to_arrays`)."""
    return CoeffEvent(
        band=np.asarray(d["band"], np.uint8), plane_gid=np.asarray(d["plane_gid"], np.int32),
        wire=np.asarray(d["wire"], np.int32), tau=np.asarray(d["tau"], np.int32),
        value=np.asarray(d["value"], np.float32),
        gids=np.asarray(d["gids"], np.int32), n_wires=np.asarray(d["n_wires"], np.int32),
        sigma_threshold=np.asarray(d["sigma_threshold"], np.float32),
        basis=basis, run=str(d["run"]), source_file=str(d["source_file"]), event=int(d["event"]),
    )


# ---- shard codec ----------------------------------------------------------

def write_coeff_shard(
    path: str | Path,
    events: list[CoeffEvent],
    *,
    dataset_name: str = "",
    file_index: int = 0,
    global_event_offset: int = 0,
    norm_sigma: np.ndarray | None = None,
    compression: str = "gzip",
) -> None:
    """Write many CoeffEvents to one flat-columnar shard. All events must share
    the basis and the plane set (gids / n_wires)."""
    if not events:
        raise ValueError("write_coeff_shard: no events")
    e0 = events[0]
    basis = e0.basis.with_derived_band_lengths() if not e0.basis.band_lengths else e0.basis
    basis.validate()
    dig = basis.digest()
    for e in events:
        if not np.array_equal(e.gids, e0.gids) or not np.array_equal(e.n_wires, e0.n_wires):
            raise ValueError("all events in a shard must share the plane set (gids/n_wires)")
        if e.basis.digest() != dig:
            raise ValueError("all events in a shard must share the basis")

    band = np.concatenate([e.band for e in events]) if events else np.empty(0, np.uint8)
    plane_gid = np.concatenate([e.plane_gid for e in events])
    wire = np.concatenate([e.wire for e in events])
    tau = np.concatenate([e.tau for e in events])
    value = np.concatenate([e.value for e in events])
    counts = np.array([e.n_coeff for e in events], dtype=np.int64)
    event_offset = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    sigma = np.stack([e.sigma_threshold for e in events]).astype(np.float32)

    with h5py.File(path, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["n_events"] = len(events)
        cfg.attrs["dataset_name"] = dataset_name
        cfg.attrs["file_index"] = file_index
        cfg.attrs["global_event_offset"] = global_event_offset
        cfg.attrs["readout_type"] = "wire"
        cfg.attrs["wavelet"] = basis.wavelet
        cfg.attrs["dwt_level"] = basis.level
        cfg.attrs["dwt_mode"] = basis.mode
        cfg.attrs["n_ticks_raw"] = basis.n_ticks_raw
        cfg.attrs["pad"] = basis.pad
        cfg.attrs["sigma_norm"] = basis.sigma_norm
        cfg.attrs["basis_digest"] = dig
        cfg.attrs["removal_json"] = json.dumps(basis.removal, sort_keys=True)
        cfg.attrs["threshold_json"] = json.dumps(basis.threshold, sort_keys=True)
        cfg.create_dataset("band_lengths", data=np.asarray(basis.band_lengths, np.int32))
        cfg.create_dataset("gids", data=e0.gids.astype(np.int32))
        cfg.create_dataset("n_wires", data=e0.n_wires.astype(np.int32))
        if norm_sigma is not None:
            cfg.create_dataset("norm_sigma", data=np.asarray(norm_sigma, np.float32))

        coord = f.create_group("coord")
        _ds(coord, "band", band, compression)
        _ds(coord, "plane_gid", plane_gid, compression)
        _ds(coord, "wire", wire, compression)
        _ds(coord, "tau", tau, compression)
        coord.create_dataset("event_offset", data=event_offset)
        _ds(coord, "sigma_threshold", sigma, compression)

        _ds(f, "value", value, compression)

        ident = f.create_group("ident")
        ident.create_dataset("run", data=np.array([e.run for e in events], dtype=_STR))
        ident.create_dataset("source_file", data=np.array([e.source_file for e in events], dtype=_STR))
        ident.create_dataset("event", data=np.array([e.event for e in events], dtype=np.int64))


def n_events(path: str | Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["config"].attrs["n_events"])


def read_coeff_event(path: str | Path, i: int) -> CoeffEvent:
    """Read event ``i`` from a shard → CoeffEvent (slices by event_offset)."""
    with h5py.File(path, "r") as f:
        cfg = f["config"]
        basis = _basis_from_config(cfg)
        off = f["coord"]["event_offset"][:]
        if i < 0 or i + 1 >= off.shape[0]:
            raise IndexError(f"event {i} out of range (n_events={off.shape[0]-1})")
        a, b = int(off[i]), int(off[i + 1])
        d = dict(
            band=f["coord"]["band"][a:b], plane_gid=f["coord"]["plane_gid"][a:b],
            wire=f["coord"]["wire"][a:b], tau=f["coord"]["tau"][a:b],
            value=f["value"][a:b],
            gids=cfg["gids"][:], n_wires=cfg["n_wires"][:],
            sigma_threshold=f["coord"]["sigma_threshold"][i],
            run=f["ident"]["run"][i].decode() if isinstance(f["ident"]["run"][i], bytes) else str(f["ident"]["run"][i]),
            source_file=f["ident"]["source_file"][i].decode() if isinstance(f["ident"]["source_file"][i], bytes) else str(f["ident"]["source_file"][i]),
            event=int(f["ident"]["event"][i]),
        )
    return arrays_to_coeff_event(d, basis)


def read_coeff_shard(path: str | Path) -> list[CoeffEvent]:
    """Read all events from a shard."""
    return [read_coeff_event(path, i) for i in range(n_events(path))]


# ---- shard audit -----------------------------------------------------------

def audit_shard(path, *, strict=True):
    """Check a coeff shard for the signatures that SILENT bugs produce.

    Structural tests (round-trip, identity, digest) all passed on a shard whose
    ``sigma_threshold`` — and therefore ``norm_sigma`` — was entirely zero,
    because nothing asserted that a field must actually *vary*. This audits for
    degenerate content: all-zero, constant-across-events, non-finite, and
    out-of-range coordinates.

    Returns a list of problem strings (empty == clean).
    """
    probs = []
    with h5py.File(path, "r") as f:
        cfg = f["config"]
        n_ev = int(cfg.attrs["n_events"])
        bl = cfg["band_lengths"][:].astype(np.int64)
        gids = cfg["gids"][:].astype(np.int64)
        nw = cfg["n_wires"][:].astype(np.int64)
        coord, val = f["coord"], f["value"][:]
        band = coord["band"][:]; pgid = coord["plane_gid"][:]
        wire = coord["wire"][:]; tau = coord["tau"][:]
        off = coord["event_offset"][:]
        sig = coord["sigma_threshold"][:]

        def bad(name, cond, msg):
            if cond:
                probs.append(f"{name}: {msg}")

        # --- degenerate content ---
        bad("value", not np.isfinite(val).all(), "non-finite entries")
        bad("value", val.size and not np.any(val != 0), "ALL ZERO")
        bad("sigma_threshold", not np.isfinite(sig).all(), "non-finite entries")
        bad("sigma_threshold", float(np.nanmax(sig)) <= 0, "ALL ZERO (norm_sigma is meaningless)")
        if n_ev > 1 and sig.size:
            bad("sigma_threshold", bool(np.all(sig == sig[0])),
                "bit-identical for every event — it is not being recomputed")
        if "norm_sigma" in cfg:
            ns = cfg["norm_sigma"][:]
            bad("norm_sigma", not np.isfinite(ns).all(), "non-finite")
            bad("norm_sigma", float(np.nanmax(ns)) <= 0, "ALL ZERO")
            bad("norm_sigma", ns.shape != (len(gids), len(bl)),
                f"shape {ns.shape} != (n_gid, n_bands) {(len(gids), len(bl))}")

        # --- offsets ---
        bad("event_offset", off.shape[0] != n_ev + 1, f"len {off.shape[0]} != n_events+1 {n_ev+1}")
        bad("event_offset", off[0] != 0, "does not start at 0")
        bad("event_offset", bool(np.any(np.diff(off) < 0)), "not monotonic")
        bad("event_offset", int(off[-1]) != val.shape[0],
            f"last {int(off[-1])} != n_values {val.shape[0]}")

        # --- coordinate ranges ---
        bad("band", band.size and (int(band.max()) >= len(bl)),
            f"max {int(band.max()) if band.size else -1} >= n_bands {len(bl)}")
        bad("plane_gid", bool(np.setdiff1d(np.unique(pgid), gids).size),
            "contains gids absent from /config/gids")
        if tau.size:
            bad("tau", bool(np.any(tau >= bl[band])), "tau >= band_lengths[band] for some row")
            bad("tau", int(tau.min()) < 0, "negative")
        if wire.size:
            g2n = {int(g): int(n) for g, n in zip(gids, nw)}
            lim = np.array([g2n[int(g)] for g in pgid], np.int64)
            bad("wire", bool(np.any(wire >= lim)), "wire >= n_wires for its plane")
            bad("wire", int(wire.min()) < 0, "negative")

        # --- identity ---
        if "ident" in f:
            evs = f["ident"]["event"][:]
            bad("ident/event", evs.shape[0] != n_ev, f"len {evs.shape[0]} != n_events {n_ev}")
            bad("ident/event", n_ev > 1 and bool(np.all(evs == evs[0])), "constant")
        else:
            probs.append("ident: missing (no event identity)")

    if strict and probs:
        raise ValueError(f"shard audit failed for {path}:\n  " + "\n  ".join(probs))
    return probs
