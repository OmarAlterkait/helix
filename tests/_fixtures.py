"""What the suite shares: the external paths it validates against, and synthetic inputs.

Sections:

* paths -- the data roots, through :mod:`helix.paths` (a collection-time skip names what
  is missing);
* a synthetic JAXTPC campaign task (``write_task``);
* a standalone coefficient-shard writer for reader tests (``write_coeff_shard``);
* plane images through :func:`helix.build.build` (``PlanesSource``, ``build_corpus``).

Each section keeps its own explanation below. Imported as ``_fixtures`` (tests/ has no
``__init__.py``; see conftest.py for why not ``tests._fixtures``). torch-free: conftest
imports it. The numpy noise reference, which needs ``helix.tpc.noise``, is
``tests/_noise_reference.py``.
"""
import os
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from helix import paths as _hpaths
from helix.core.coeff_io import coord_digest
from helix.build import build
from helix.core.coeff_io import read_coeff_event, read_coeff_shard


# ---- paths -------------------------------------------------------------------------------

# External paths the suite validates against, in one place and overridable.
#
# Hardcoded absolute paths spread over test files are quiet in two ways: a test
# can keep validating against an unversioned external tree without anyone
# noticing, and on a machine without the path it skips, and a skip for "the data is
# not here" looks exactly like a skip for "this machine has no GPU". Nothing
# distinguishes a suite that checked nothing from one that checked everything.
#
# pimm-data solved this with ``*_DATA_ROOT`` env vars plus a collection-time skip;
# this mirrors it. Defaults preserve today's behaviour exactly, so nothing changes
# unless an env var is set.
#
#     HELIX_PIMM_ROOT       the pimm checkout the config/step contracts import
#     HELIX_CORPUS          a built coeff corpus
#     HELIX_SIM_ROOT        the JAXTPC wire campaign (<campaign>/data/jaxtpc_wire)
#

def _opt(name):
    """A root's value, or "" when this site does not configure it.

    ALWAYS via helix.paths, never a second default. Every entry here used to
    carry its own copy of an S3DF literal, and one of them had already drifted
    (see CORPUS below). "" rather than None so the `os.path.isdir(...)` guards
    that gate every real-data test keep working unchanged -- isdir("") is False.

    `resolve`, not `root`: root() RAISES on an unconfigured root, and this module
    is imported at COLLECTION time. A raise here would turn "this machine has no
    corpus" into an error that hides the entire suite, which is the opposite of
    what these guards are for.
    """
    value, _src = _hpaths.resolve(name)
    return str(value) if value is not None else ""


#: pimm checkout providing the trainer/registry the contract tests import.
PIMM_ROOT = _opt("HELIX_PIMM_ROOT")

#: A built coefficient corpus (noisy + clean shard families).
#
#: Resolved by :mod:`helix.paths`, not redeclared here. This used to carry its
#: OWN default -- and a different one: the pre-tau `coeff_tpc` vintage, while
#: helix.paths said `coeff_tpc_r1`. One env var, two values, and two KINDS of
#: value (a run dir here, a corpus root there). Both corpora are real, both read
#: cleanly, and they differ in the coherent-removal gate (r1 records tau=0.05),
#: so the wrong one is a plausible wrong answer rather than a crash.
CORPUS = _opt("HELIX_CORPUS")

#: The JAXTPC wire campaign the corpus is built from. Each test picks its own task beneath
#: it, so this is a ROOT, not one file.
SIM_ROOT = _opt("HELIX_SIM_ROOT")

#: The task the real-data tests read: it holds an event with an empty volume (entry 202).
SIM_GROUP, SIM_TASK = "j000000", "jaxtpc_wire_j000085-000089"


def sim_file(kind="sensor", group=SIM_GROUP, task=SIM_TASK):
    """One task's ``kind`` (sensor / step / hits / labl) file beneath :data:`SIM_ROOT`."""
    return os.path.join(SIM_ROOT, group, task, kind, f"sim_wire_{kind}_{task}.h5")


def describe():
    """``{name: (path, exists)}`` — so a skip can say WHICH input was missing."""
    return {n: (p, os.path.exists(p)) for n, p in (
        ("PIMM_ROOT", PIMM_ROOT), ("CORPUS", CORPUS),
        ("SIM_ROOT", SIM_ROOT))}


# ---- a synthetic JAXTPC campaign task ----------------------------------------------------

# A synthetic JAXTPC wire campaign task: ``sensor``, ``step`` and ``hits`` files in the layout and
# format of ``doraemon_2026_1M_lartpc`` (format 2, helix/tpc/io.py), and the task's pysupera ``labl``
# (pimm-data ``supera_labl``), small enough for a test.
#
#     <root>/<group>/<task>/{sensor,step,hits,labl}/sim_wire_{kind}_<task>.h5
#
# Every file has ``/config`` (format_version 2, production_version, git_commit), ``/dprod/events``
# and ``entry_NNNNNN`` groups carrying the campaign identity (event_id, job_id, job_event). The sensor
# holds per volume and plane a sparse digitized readout (delta-encoded COO, raw ADC = pedestal +
# signal); ``step`` per deposit ``de``, ``charge`` and quantised ``positions`` (uint16 *
# ``pos_step_mm`` + ``pos_origin_{x,y,z}``); ``hits`` per plane compact particles (tracks) of one to
# three groups each, scattered over the plane, some placed on top of one another so cells hold charge
# from several tracks; one group's samples may land on the same pixel more than once.
#
# An (entry, volume) listed in ``empty`` has nothing deposited, written as the simulator writes it:
# sensor plane groups with no datasets and no attrs; hits ``n_actual = 0``, planes with no entries,
# ``group_to_track`` holding only the null group, no per-deposit arrays; a step volume with
# ``n_actual = 0`` and no datasets.
#

CAMPAIGN = "test_campaign"
N_TICKS = 64
PLANES = ("U", "V", "Y")
N_WIRES = {"U": 40, "V": 40, "Y": 30}
PEDESTAL = {"U": 1843, "V": 1843, "Y": 410}
EVENTS_PER_JOB = 200


def task_name(first_job, n_jobs=5):
    return f"jaxtpc_wire_j{first_job:06d}-{first_job + n_jobs - 1:06d}"


def _config(f, n_volumes, kind):
    cfg = f.create_group("config")
    cfg.attrs.update(format_version=2, production_version=CAMPAIGN, git_commit="abc", git_dirty=False,
                     readout_type="wire", n_volumes=n_volumes)
    cfg["num_wires"] = np.array([[N_WIRES[p] for p in PLANES]] * n_volumes, np.int32)
    if kind == "sensor":
        cfg.attrs["num_time_steps"] = N_TICKS
        cfg["pedestals"] = np.array([[PEDESTAL[p] for p in PLANES]] * n_volumes, np.int32)


def _hits_plane(rng, centres):
    """One hits plane group's arrays from ``[(group_id, cw, ct)]``."""
    cw, ct, gid, sz, pk, dw, dt, qs = [], [], [], [], [], [], [], []
    for g, w, t in centres:
        n = int(rng.integers(3, 40))
        cw.append(w); ct.append(t); gid.append(g); sz.append(n); pk.append(rng.uniform(50, 2000))
        dw.append(rng.integers(-3, 4, n)); dt.append(rng.integers(-12, 13, n)); qs.append(rng.integers(0, 65536, n))
    return dict(center_wires=np.array(cw, np.int16), center_times=np.array(ct, np.int16),
                group_ids=np.array(gid, np.int32), group_sizes=np.array(sz, np.uint16),
                peak_charges=np.array(pk, np.float32), delta_wires=np.concatenate(dw).astype(np.int8),
                delta_times=np.concatenate(dt).astype(np.int8), charges_u16=np.concatenate(qs).astype(np.uint16))


def _sensor_plane(rng, g, p):
    """A sparse plane: sorted unique (wire, tick), signal >= 3 ADC; returns its dense truth."""
    flat = np.sort(rng.choice(N_WIRES[p] * N_TICKS, size=25, replace=False))
    wire, time = np.divmod(flat, N_TICKS)
    signal = rng.integers(3, 300, size=flat.size)
    g["delta_wire"] = np.diff(wire, prepend=wire[0]).astype(np.int16)
    g["delta_time"] = np.diff(time, prepend=time[0]).astype(np.int16)
    g["values"] = (signal + PEDESTAL[p]).astype(np.uint16)
    g.attrs.update(wire_start=int(wire[0]), time_start=int(time[0]), pedestal=PEDESTAL[p], n_pixels=flat.size)
    dense = np.zeros((N_WIRES[p], N_TICKS), np.float32)
    dense[wire, time] = signal
    return dense


def write_task(root, group="j000000", first_job=0, n_events=3, seed=0, n_volumes=2, n_tracks=30, empty=()):
    """One task of jobs ``first_job ..`` holding campaign events ``first_job * 200 ..``.

    Returns ``dict(sensor=, step=, hits=, truth={(entry, label): dense pedestal-subtracted image})``.
    """
    rng = np.random.default_rng(seed)
    task = task_name(first_job)
    paths = {k: os.path.join(root, group, task, k, f"sim_wire_{k}_{task}.h5") for k in ("sensor", "step", "hits")}
    for p in paths.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    first = first_job * EVENTS_PER_JOB
    ids = np.arange(first, first + n_events, dtype=np.int64)
    truth = {}
    with h5py.File(paths["sensor"], "w") as fe, h5py.File(paths["hits"], "w") as fh, \
            h5py.File(paths["step"], "w") as fs:
        for kind, f in (("sensor", fe), ("hits", fh), ("step", fs)):
            _config(f, n_volumes, kind)
            ev = f.create_group("dprod").create_group("events")
            ev.attrs["order"] = "sorted"
            ev["event_id"], ev["job_id"], ev["job_event"] = ids, ids // EVENTS_PER_JOB, ids % EVENTS_PER_JOB
        for pos, eid in enumerate(ids.tolist()):
            ee, eh, es = (f.create_group(f"entry_{pos:06d}") for f in (fe, fh, fs))
            for g in (ee, eh, es):
                g.attrs.update(event_id=eid, job_id=eid // EVENTS_PER_JOB, job_event=eid % EVENTS_PER_JOB,
                               source_index=(eid - first) // EVENTS_PER_JOB,
                               source_event_idx=eid % EVENTS_PER_JOB, n_volumes=n_volumes)
            eh.attrs["threshold"] = 25.0
            for v in range(n_volumes):
                sv_e = ee.create_group(f"volume_{v}")
                hv, sv = eh.create_group(f"volume_{v}"), es.create_group(f"volume_{v}")
                if (pos, v) in empty:
                    for p in PLANES:
                        sv_e.create_group(p)
                        truth[(pos, f"volume_{v}_{p}")] = np.zeros((N_WIRES[p], N_TICKS), np.float32)
                    hv.attrs.update(n_actual=0, n_groups=1); sv.attrs["n_actual"] = 0
                    hv["group_to_track"] = np.zeros(1, np.int32)
                    for p in PLANES:
                        hv.create_group(p).attrs.update(n_entries=0, n_groups_plane=0)
                    continue
                for p in PLANES:
                    truth[(pos, f"volume_{v}_{p}")] = _sensor_plane(rng, sv_e.create_group(p), p)
                n_groups = 0
                g2t, centres = [], {p: [] for p in PLANES}
                for trk in range(n_tracks):
                    tid = int(rng.integers(0, 10_000))
                    if trk % 5 == 4 and centres["U"]:                 # on top of the previous track
                        base = {p: centres[p][-1][1:] for p in PLANES}
                    else:
                        base = {p: (int(rng.integers(40, 900)), int(rng.integers(200, 3000))) for p in PLANES}
                    for _ in range(int(rng.integers(1, 4))):
                        for p in PLANES:
                            centres[p].append((n_groups, base[p][0] + int(rng.integers(-2, 3)),
                                               base[p][1] + int(rng.integers(-8, 9))))
                        g2t.append(tid); n_groups += 1
                n_dep = 4 * n_groups
                hv.attrs.update(n_actual=n_dep, n_groups=n_groups)
                sv.attrs.update(n_actual=n_dep, pos_step_mm=0.3, pos_origin_x=float(rng.uniform(-2000, 0)),
                                pos_origin_y=float(rng.uniform(-2000, 0)), pos_origin_z=float(rng.uniform(-2000, 0)))
                hv["group_to_track"] = np.array(g2t, np.int32)
                hv["deposit_to_group"] = rng.integers(0, n_groups, n_dep).astype(np.int32)
                sv["de"] = rng.uniform(0, 0.05, n_dep).astype(np.float16)
                sv["charge"] = rng.uniform(0, 5000, n_dep).astype(np.float32)
                sv["positions"] = rng.integers(0, 13000, (n_dep, 3)).astype(np.uint16)
                for p in PLANES:
                    pg = hv.create_group(p)
                    for k, a in _hits_plane(rng, centres[p]).items():
                        pg[k] = a
    paths["labl"] = _write_labl(root, group, task, ids, fh_path=paths["hits"], seed=seed)
    return dict(paths, truth=truth)


#: pdg codes the fixture's particles draw from: the five PID classes (one sign-flipped) and one
#: outside them (a kaon -> pid -1)
PDGS = (22, 11, 13, -13, 211, 2212, 321)


def _write_labl(root, group, task, ids, fh_path, seed):
    """The task's labl: every hits track listed in ``track/``, owned by its own fragment row except
    every 7th (fragment -1); each fragment its own instance, semantic types and PDGs drawn."""
    rng = np.random.default_rng(seed + 1)
    path = os.path.join(root, group, task, "labl", f"sim_wire_labl_{task}.h5")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(fh_path, "r") as fh, h5py.File(path, "w") as f:
        f.create_group("config").attrs.update(labl_format="supera-labl/1", readout_type="wire")
        ev = f.create_group("dprod").create_group("events")
        ev["event_id"], ev["job_id"], ev["job_event"] = ids, ids // EVENTS_PER_JOB, ids % EVENTS_PER_JOB
        for pos, eid in enumerate(ids.tolist()):
            src = fh[f"entry_{pos:06d}"]
            tids = np.unique(np.concatenate([src[v]["group_to_track"][:] for v in src if v.startswith("volume_")]))
            n = len(tids)
            e = f.create_group(f"entry_{pos:06d}")
            e.attrs.update(event_id=eid, job_id=eid // EVENTS_PER_JOB, job_event=eid % EVENTS_PER_JOB)
            frag = np.arange(n, dtype=np.int32)
            frag[::7] = -1
            e["track/id"], e["track/fragment"] = tids.astype(np.int32), frag
            e["particle/fragment_sem_type"] = rng.integers(1, 6, n).astype(np.int8)
            e["particle/instance_sem_type"] = rng.integers(1, 6, n).astype(np.int8)
            e["particle/fragment_instance"] = np.arange(n, dtype=np.int32)
            e["particle/pdg"] = rng.choice(PDGS, n).astype(np.int32)
    return path


def write_geometry(path, n_volumes=2):
    """A plane registry (helix/tpc/readout.py) of the fixture's planes."""
    import json
    planes = {f"volume_{v}_{p}": dict(n_wires=N_WIRES[p], pedestal=PEDESTAL[p], wire_lengths_m=2.0)
              for v in range(n_volumes) for p in PLANES}
    with open(path, "w") as fh:
        json.dump(dict(detector="test", num_time_steps=N_TICKS, sampling_rate_hz=2e6, planes=planes), fh)
    return str(path)


# ---- coefficient shards for reader tests -------------------------------------------------

# A standalone writer of coefficient shards from per-event dicts, for reader tests: it writes
# what helix.core.coeff_io writes, and also shards the codec would refuse (to test that the
# reader refuses them).

_STR = h5py.string_dtype(encoding="utf-8")


def basis_digest(*, wavelet, dwt_level, dwt_mode, n_ticks_raw, pad, band_lengths,
                 removal_json="{}", threshold_json="{}", sigma_norm=2.6) -> str:
    """sha256 of the canonical basis dict — mirrors ``BasisDescriptor.digest()``.

    Field NAMES are the descriptor's (``level``/``mode``), not the ``/config``
    attr names (``dwt_level``/``dwt_mode``). helix re-derives this on every read
    and refuses the shard on mismatch, so pimm-data computes it rather than
    trusting a caller-supplied string — a wrong digest produced a shard that
    pimm-data wrote happily and helix rejected. Pinned by the cross-repo golden.
    """
    d = dict(wavelet=str(wavelet), level=int(dwt_level), mode=str(dwt_mode),
             n_ticks_raw=int(n_ticks_raw), pad=int(pad),
             band_lengths=[int(x) for x in band_lengths],
             removal=json.loads(str(removal_json) or "{}"),
             threshold=json.loads(str(threshold_json) or "{}"),
             sigma_norm=float(sigma_norm))
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ds(grp, name, data, comp):
    kw = dict(compression=comp, compression_opts=4, shuffle=True) if comp == "gzip" else {}
    grp.create_dataset(name, data=data, **kw)


def write_coeff_shard(path, events, *, band_lengths, gids, n_wires, basis_attrs,
                      norm_sigma=None, dataset_name="", file_index=0,
                      global_event_offset=0, compression="gzip",
                      coords=True, noise=None, provenance=None):
    """Write a flat-columnar coeff shard (standalone; the layout helix reads).

    ``events`` is a list of per-event dicts with keys ``band, plane_gid, wire,
    tau, value, sigma_threshold, run, source_file, event``. All events share the
    plane set (``gids`` / ``n_wires``) and the basis. ``basis_attrs`` supplies the
    ``/config`` scalar attrs (wavelet, dwt_level, dwt_mode, n_ticks_raw, pad,
    sigma_norm, basis_digest, removal_json, threshold_json).

    ``coords=False`` writes a values-only shard for a co-supported target (see
    :func:`coord_digest`); ``noise`` records how the input was noised, so two
    shards built with different noise are distinguishable after the fact.

    ``provenance`` writes ``/config@provenance_json`` — the ``code`` block and the
    ``geom``/``spectrum`` hashes that :mod:`pimm_data.coeff_verify` compares
    across a corpus. This writer omitted it entirely, so no shard it produced
    could exercise those checks, and every fixture-based test of the verifier was
    silently testing the legacy path.
    """
    if not events:
        raise ValueError("write_coeff_shard: no events")

    # validate the basis contract so a pimm-written shard is not silently malformed
    # (helix's reader validates on read; fail here at write time instead).
    _REQUIRED = ("wavelet", "dwt_level", "dwt_mode", "n_ticks_raw", "pad", "sigma_norm")
    missing = [k for k in _REQUIRED if k not in basis_attrs]
    if missing:
        raise ValueError(f"basis_attrs missing required keys {missing}")
    # helix's reader re-hashes the basis and json-parses these on every read, so
    # anything it would reject must fail HERE — otherwise pimm-data happily writes
    # shards that the reference codec refuses, and the cross-repo contract is only
    # enforced in one direction.
    for k in ("removal_json", "threshold_json"):
        if k in basis_attrs:
            try:
                json.loads(str(basis_attrs[k]))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"basis_attrs[{k!r}] is not parseable JSON: {e}") from e
    import pywt
    padded = int(basis_attrs["n_ticks_raw"]) + int(basis_attrs["pad"])
    want = tuple(int(c.shape[-1]) for c in pywt.wavedec(
        np.zeros(padded, np.float32), str(basis_attrs["wavelet"]),
        level=int(basis_attrs["dwt_level"]), mode=str(basis_attrs["dwt_mode"])))
    if tuple(int(x) for x in band_lengths) != want:
        raise ValueError(
            f"band_lengths {tuple(int(x) for x in band_lengths)} inconsistent with basis "
            f"({basis_attrs['wavelet']} L{basis_attrs['dwt_level']} @ {padded}) → expected {want}")

    dig = basis_digest(
        wavelet=basis_attrs["wavelet"], dwt_level=basis_attrs["dwt_level"],
        dwt_mode=basis_attrs["dwt_mode"], n_ticks_raw=basis_attrs["n_ticks_raw"],
        pad=basis_attrs["pad"], band_lengths=band_lengths,
        removal_json=basis_attrs.get("removal_json", "{}"),
        threshold_json=basis_attrs.get("threshold_json", "{}"),
        sigma_norm=basis_attrs["sigma_norm"])
    given = basis_attrs.get("basis_digest")
    if given is not None and str(given) != dig:
        raise ValueError(
            f"basis_digest {str(given)!r} does not match the basis it describes "
            f"(hashes to {dig!r}). helix's reader recomputes this and refuses the "
            f"shard, so writing it would produce a file only pimm-data can read.")
    basis_attrs = {**basis_attrs, "basis_digest": dig}

    def cat(key, dt):
        parts = [np.asarray(e[key], dt) for e in events]
        return np.concatenate(parts) if parts else np.empty(0, dt)

    band = cat("band", np.uint8)
    plane_gid = cat("plane_gid", np.int32)          # int32: gid can exceed 255 (matches helix)
    wire = cat("wire", np.int32)
    tau = cat("tau", np.int32)
    value = cat("value", np.float32)
    counts = np.array([len(e["value"]) for e in events], np.int64)
    event_offset = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    sigma = np.stack([np.asarray(e["sigma_threshold"], np.float32) for e in events])
    digests = np.array([coord_digest(e["band"], e["plane_gid"], e["wire"], e["tau"])
                        for e in events], np.uint64)

    with h5py.File(path, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["format"] = "helix-coeff-2"
        cfg.attrs["n_events"] = len(events)
        cfg.attrs["dataset_name"] = dataset_name
        cfg.attrs["file_index"] = file_index
        cfg.attrs["global_event_offset"] = global_event_offset
        cfg.attrs["readout_type"] = "wire"
        for k, v in basis_attrs.items():
            cfg.attrs[k] = v
        cfg.attrs["has_coords"] = bool(coords)
        cfg.attrs["noise_json"] = json.dumps(noise or {}, sort_keys=True)
        if provenance is not None:
            cfg.attrs["provenance_json"] = json.dumps(provenance, sort_keys=True)
        cfg.create_dataset("band_lengths", data=np.asarray(band_lengths, np.int32))
        cfg.create_dataset("gids", data=np.asarray(gids, np.int32))
        cfg.create_dataset("n_wires", data=np.asarray(n_wires, np.int32))
        if norm_sigma is not None:
            cfg.create_dataset("norm_sigma", data=np.asarray(norm_sigma, np.float32))

        coord = f.create_group("coord")
        if coords:
            _ds(coord, "band", band, compression)
            _ds(coord, "plane_gid", plane_gid, compression)
            _ds(coord, "wire", wire, compression)
            _ds(coord, "tau", tau, compression)
        coord.create_dataset("event_offset", data=event_offset)
        coord.create_dataset("coord_digest", data=digests)
        _ds(coord, "sigma_threshold", sigma, compression)

        _ds(f, "value", value, compression)

        ident = f.create_group("ident")
        ident.create_dataset("run", data=np.array([e.get("run", "") for e in events], dtype=_STR))
        ident.create_dataset("source_file", data=np.array([e.get("source_file", "") for e in events], dtype=_STR))
        ident.create_dataset("event", data=np.array([e.get("event", -1) for e in events], np.int64))


# ---- plane images through helix.build ----------------------------------------------------

# Plane images from a function through :func:`helix.build.build`, for tests: a source with no
# reader, so a test controls the input exactly.

class PlanesSource:
    """``events``: ints, or (source_file, event[, noise_seed]); ``plane_fn(event) -> (noisy, clean)``
    as ``{gid: (wires, ticks)}`` images (``clean`` None or {} for no clean target)."""

    systems = ()

    def __init__(self, events, plane_fn, *, run="", noise=None, provenance=None):
        self.events_, self.plane_fn, self.run = list(events), plane_fn, run
        self.attrs = dict(noise=noise, provenance=provenance)

    def __len__(self):
        return len(self.events_)

    def bind(self, chain):
        pass

    def shard_attrs(self):
        return dict(self.attrs)

    def events(self, indices, workers):
        for k in indices:
            e = self.events_[k]
            src, ev, seed = (e[0], e[1], e[2] if len(e) > 2 else -1) if isinstance(e, (tuple, list)) else ("", e, -1)
            noisy, clean = self.plane_fn(ev)
            ident = dict(run=self.run, source_file=src, event=int(ev))
            if seed >= 0:
                ident["noise_seed"] = int(seed)
            yield dict(ident=ident, planes=noisy, clean=clean or None)


def build_corpus(events, plane_fn, chain, out_dir, *, dataset_name="cx", run="", norm_sigma=None,
                 with_clean=True, noise=None, provenance=None):
    """Build one shard ``out_dir/<dataset_name>_coeff_0000.h5`` (+ clean); returns the events
    read back (noisy, clean) and the shard's norm_sigma."""
    fn = plane_fn if with_clean else (lambda ev: (plane_fn(ev)[0], None))
    src = PlanesSource(events, fn, run=run, noise=noise, provenance=provenance)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    build(src, str(Path(out_dir) / dataset_name), indices=list(range(len(src))), chain=chain,
          events_per_shard=len(src), norm_sigma=norm_sigma, keep_clean=with_clean)
    p = Path(out_dir) / f"{dataset_name}_coeff_0000.h5"
    noisy = read_coeff_shard(p)
    pc = Path(out_dir) / f"{dataset_name}_coeff_clean_0000.h5"
    clean = [read_coeff_event(pc, i, coords_from=p) for i in range(len(noisy))] if pc.exists() else []
    with h5py.File(p, "r") as f:
        norm = f["config"]["norm_sigma"][:]
    return noisy, clean, norm
