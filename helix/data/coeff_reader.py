"""Coeff-corpus reader — the flat-columnar wavelet-coefficient shards (helix.core.coeff_io writes them).

The coeff corpus is a purpose-built training cache (produced by helix DSP), read
hot in the loop, so it is **flat columnar**, not per-event groups: shard-wide
``/coord`` + ``/value`` arrays plus an ``event_offset`` index; one contiguous
slice per event; ``plane_gid`` is a column. This is pimm-data's own offset idiom
(cf. the optical second row-space) applied shard-wide.

One modality per file (``{name}_coeff_*.h5`` noisy input, ``{name}_coeff_clean_*``
the noise-free values of a ``--keep-clean`` build), joined by identity — never mixed. The on-disk layout is the
shared contract defined in helix ``docs/archive/COEFF_CORPUS_DESIGN.md`` §3; this reader
implements it with **standalone h5py** (no helix import at read time — no cycle),
and is round-trip golden-tested against helix's reference codec.

Layout::

    /config  attrs: n_events, dataset_name, file_index, global_event_offset,
                    readout_type, wavelet, dwt_level, dwt_mode, n_ticks_raw, pad,
                    sigma_norm, basis_digest, removal_json, threshold_json
             datasets: band_lengths (n_bands,), gids (G,), n_wires (G,),
                       [norm_sigma (G, n_bands)]
    /coord   band uint8, plane_gid uint8, wire int32, tau int32   (all (M,))
             event_offset int64 (n_events+1,); sigma_threshold f32 (n_events,G,n_bands)
    /value   float32 (M,)
    /ident   run (n_events,), source_file (n_events,), event int64 (n_events,)
"""

import glob
import hashlib
import os

import numpy as np
import h5py

from pimm_data import ShardReaderBase

from helix.core.coeff_io import FORMAT, coord_digest, shard_format


class CoeffTPCReader(ShardReaderBase):
    """Flat-columnar coeff-shard reader. ``modality`` selects the file family
    (``'coeff'`` noisy / ``'coeff_clean'`` noise-free values)."""

    _MODALITY = "coeff"

    def __init__(self, data_root, split="", dataset_name="coeff_tpc", modality="coeff", **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self._MODALITY = modality               # instance attr shadows the class default
        # Shard-shared metadata. Taken from the first shard AND verified against
        # every other one: these tables are corpus-wide by contract (norm_sigma
        # in particular must be frozen across shards), so a disagreement means
        # the shards are not one corpus. Serving shard 0's tables for every event
        # silently mis-normalises everything outside shard 0.
        self.band_lengths = None
        self.gids = None
        self.n_wires = None
        self.norm_sigma = None
        self.sigma_norm = None
        self.config_attrs = None
        self.has_coords = True
        self._meta_source = None
        self._meta_by_path = {}                 # h5_path -> its /config tables
        self._ident_by_path = {}                # h5_path -> raw (run, source_file, event)
        self._dup_ident = {}                    # h5_path -> (rows, unique) when not 1:1
        self._bad_format = {}                   # h5_path -> its format when not FORMAT
        self._id2pos = {}                       # h5_path -> {physical event id: write-order position}
        self._init_shards()
        self._validate_shard_agreement()

    def _validate_shard_agreement(self):
        """Every shard of one corpus must share the basis, plane set and — above
        all — ``norm_sigma``. Row i of that table normalises any event in any
        shard, so a per-shard table means the same physical coefficient is scaled
        differently depending on where it was written.

        Deliberately run OUTSIDE ``_build_index``: that loop catches every
        exception and downgrades it to a log warning, dropping the offending
        shard's events silently — so raising from ``_index_for_shard`` would turn
        a mis-built corpus into a quietly shorter one.
        """
        def _as_text(v):
            """Bytes and str compare equal — h5py returns either depending on
            how the attr was written, and a spurious mismatch here would refuse
            a perfectly good corpus."""
            if v is None:
                return None
            if isinstance(v, bytes):
                return v.decode("utf-8", "replace")
            return str(v)

        def _same(a, b):
            if a is None or b is None:
                return a is None and b is None
            return np.array_equal(np.asarray(a), np.asarray(b))

        for path, fmt in self._bad_format.items():
            raise ValueError(f"{path}: format {fmt!r}, not {FORMAT!r}; rebuild it or migrate it "
                             f"(scripts/migrate_format.py)")
        for path, (rows, uniq) in self._dup_ident.items():
            raise ValueError(
                f"{path}: /ident does not uniquely identify its events — {rows} rows "
                f"collapse to {uniq} identities. (run, source_file, event) must be "
                f"unique within a shard; duplicates would be silently dropped.")
        if not self._meta_by_path:
            return
        (ref_path, ref), *rest = self._meta_by_path.items()
        for path, m in rest:
            for name in ("band_lengths", "gids", "n_wires", "norm_sigma"):
                if not _same(ref[name], m[name]):
                    a = None if ref[name] is None else np.asarray(ref[name]).tolist()
                    b = None if m[name] is None else np.asarray(m[name]).tolist()
                    raise ValueError(
                        f"{self._MODALITY} shards disagree on /config/{name}:\n"
                        f"  {ref_path} -> {a}\n  {path} -> {b}\n"
                        f"These shards are not one corpus. norm_sigma in particular must "
                        f"be FROZEN across a corpus (build with --norm-sigma).")
            if abs(float(m["sigma_norm"]) - float(ref["sigma_norm"])) > 1e-6:
                raise ValueError(
                    f"{self._MODALITY} shards disagree on sigma_norm: "
                    f"{ref_path} -> {ref['sigma_norm']}, {path} -> {m['sigma_norm']}")
            if bool(m["has_coords"]) != bool(ref["has_coords"]):
                raise ValueError(
                    f"{self._MODALITY} shards disagree on has_coords: "
                    f"{ref_path} -> {ref['has_coords']}, {path} -> {m['has_coords']}")
            # The DSP identity. Everything above is geometry and normalisation,
            # which two corpora built with DIFFERENT signal processing share
            # exactly — so none of it would have caught shards from two gates
            # landing in one directory. That is not hypothetical: the corpus
            # submit script defaulted its output root to the older corpus while
            # defaulting its gate to the newer one, so the ordinary way to
            # extend a corpus wrote r1-gated shards into the pre-r1 tree, where
            # they would have read as one corpus and trained as two.
            for name, what in (("basis_digest", "wavelet basis / threshold"),
                               ("removal_json", "coherent-removal gate")):
                a, b = ref.get(name), m.get(name)
                if _as_text(a) != _as_text(b):
                    raise ValueError(
                        f"{self._MODALITY} shards disagree on /config/{name} "
                        f"({what}):\n  {ref_path} -> {_as_text(a)}\n"
                        f"  {path} -> {_as_text(b)}\n"
                        f"These shards were built with different DSP and are not "
                        f"one corpus, however similar their geometry looks.")

    def _find_files(self):
        """Glob shards with the tag anchored to a digit — ``{name}_{modality}_[0-9]*.h5``
        — so ``modality='coeff'`` does NOT also match ``{name}_coeff_clean_*.h5``
        (``coeff`` is a prefix of ``coeff_clean``)."""
        pat = f"{self.dataset_name}_{self._MODALITY}_[0-9]*.h5"
        if isinstance(self.split, (list, tuple)):
            files, file_runs = [], []
            for run in self.split:
                rf = sorted(glob.glob(os.path.join(self.data_root, run, pat)))
                if not rf:
                    raise FileNotFoundError(
                        f"runs=: no {self._MODALITY} shards for run {run!r} under {self.data_root}")
                files.extend(rf); file_runs.extend([run] * len(rf))
            self._file_runs = file_runs
            return files
        for p in (os.path.join(self.data_root, self.split, pat),
                  os.path.join(self.data_root, pat)):
            files = sorted(glob.glob(p))
            if files:
                return files
        return []

    def _index_for_shard(self, h5_path):
        """Present event **ids** for one flat shard — the ``/ident/event`` physical
        ids (in write order) so the cross-modality joint index aligns coeff↔coeff_clean
        by IDENTITY, not position. Also caches shared basis/geometry, and a
        physical-id→write-position map (event_offset is positional)."""
        with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
            cfg = f["config"]
            # Recorded, not raised, for the reason given at _dup_ident below.
            if shard_format(cfg) != FORMAT:
                self._bad_format[str(h5_path)] = shard_format(cfg)
                return np.zeros(0, np.int64)
            bl = cfg["band_lengths"][:].astype(np.int32)
            gids = cfg["gids"][:].astype(np.int32)
            n_wires = cfg["n_wires"][:].astype(np.int32)
            ns = cfg["norm_sigma"][:].astype(np.float32) if "norm_sigma" in cfg else None
            sn = float(cfg.attrs["sigma_norm"])
            hc = bool(cfg.attrs["has_coords"])
            self._meta_by_path[str(h5_path)] = dict(
                band_lengths=bl, gids=gids, n_wires=n_wires, norm_sigma=ns,
                sigma_norm=sn, has_coords=hc,
                # The DSP identity, as distinct from the geometry: two corpora can
                # share every table above and differ only in the coherent gate.
                basis_digest=cfg.attrs.get("basis_digest"),
                removal_json=cfg.attrs.get("removal_json"))
            if self.band_lengths is None:
                self.config_attrs = dict(cfg.attrs)
                self.band_lengths, self.gids, self.n_wires = bl, gids, n_wires
                self.norm_sigma, self.sigma_norm = ns, sn
                self.has_coords = hc
                self._meta_source = str(h5_path)
            ev = f["ident"]["event"][:].astype(np.int64)
            src = f["ident"]["source_file"][:]
            rn = f["ident"]["run"][:]
            # Keep the RAW triple as well as the packed id. The packed id
            # exists to align coeff<->coeff_clean and is deliberately lossy
            # (a hash, once a corpus spans several runs); it cannot be turned
            # back into a source file and event number. Anything that wants
            # to join a coeff event to the simulation event it came from —
            # a probe target, a truth label — needs the triple itself.
            self._ident_by_path[str(h5_path)] = dict(
                run=_as_str_list(rn, len(ev)),
                source_file=_as_str_list(src, len(ev)),
                event=ev.copy())
            ev = _identity_ids(src, ev, run=rn)
            # Record, don't raise: _build_index wraps this call in a bare
            # `except Exception` that downgrades any error to a log warning and
            # serves the shard as EMPTY — so raising here would turn a
            # detectable fault into a silently shorter dataset. Checked in
            # _validate_shard_agreement, which runs outside that loop.
            if len(np.unique(ev)) != ev.shape[0]:
                self._dup_ident[str(h5_path)] = (ev.shape[0], len(np.unique(ev)))
        # keyed by path (not shard index) so it survives build_joint_index re-selection
        self._id2pos[str(h5_path)] = {int(e): i for i, e in enumerate(ev)}
        return np.asarray(ev, dtype=np.int64)

    def _locate_flat(self, idx):
        self._ensure_open()
        file_idx, event_id = self.locate(idx)           # event_id = physical /ident/event id
        path = str(self.h5_files[file_idx])
        pos = self._id2pos[path][int(event_id)]
        return self._h5data[file_idx], pos, path

    def event_sizes(self):
        """Coefficients per event, aligned with this reader's global index.

        Read from ``/coord/event_offset`` alone -- one small ``(n_events+1,)``
        int64 array per shard -- so the whole corpus is sized without touching a
        single coefficient. 150k events cost a few seconds and a few megabytes.

        ``CoeffTPCDataset``'s ``max_event_size`` cap reads it; anything else that
        reasons about cost per event -- a memory estimate, a shard-balance check --
        wants the same table.
        """
        self._ensure_open()
        n = len(self)
        out = np.empty(n, dtype=np.int64)
        offsets = {}
        for i in range(n):
            f, pos, path = self._locate_flat(i)
            off = offsets.get(path)
            if off is None:
                off = f["coord"]["event_offset"][:]
                offsets[path] = off
            out[i] = int(off[pos + 1]) - int(off[pos])
        return out

    def read_event(self, idx):
        """One event → flat dict of coeff rows (sliced by ``event_offset`` at the
        physical event's write position).

        A values-only shard (``has_coords=False`` — the co-supported clean
        target) yields ``value`` and ``_coord_digest`` but no coordinate columns;
        :class:`~pimm_data.coeff.CoeffTPCDataset` supplies them from the paired
        noisy event after checking the digest.
        """
        f, ev, path = self._locate_flat(idx)
        coord = f["coord"]
        off = coord["event_offset"]
        a, b = int(off[ev]), int(off[ev + 1])
        out = {"value": f["value"][a:b].astype(np.float32)}
        out["_coord_digest"] = int(coord["coord_digest"][ev])
        if "wire_valid" in coord:                    # packed (G, ceil(max n_wires / 8))
            out["_wire_valid"] = coord["wire_valid"][ev]
        ident = self._ident_by_path.get(path)
        if ident is not None:
            out["_ident"] = dict(run=ident["run"][ev],
                                 source_file=ident["source_file"][ev],
                                 event=int(ident["event"][ev]))
        if self.has_coords:
            out.update(
                band=coord["band"][a:b].astype(np.int32),
                plane_gid=coord["plane_gid"][a:b].astype(np.int32),
                wire=coord["wire"][a:b].astype(np.int32),
                tau=coord["tau"][a:b].astype(np.int32),
            )
        return out



class CellsReader(ShardReaderBase):
    """The simulation-truth family ``{name}_cells_<k>.h5`` (:mod:`helix.data.cells`), indexed
    by the same identity as the coeff shards so the joint index pairs them event by event."""

    _MODALITY = "cells"
    _find_files = CoeffTPCReader._find_files            # digit-anchored glob

    def __init__(self, data_root, split="", dataset_name="coeff_tpc"):
        self.data_root, self.split, self.dataset_name = data_root, split, dataset_name
        self._ident_by_path, self._id2pos, self._bad = {}, {}, {}
        self._init_shards()
        for path, why in self._bad.items():             # recorded, not raised: see _dup_ident above
            raise ValueError(why)

    def _index_for_shard(self, h5_path):
        from helix.data.cells import read_ident, require_format
        with h5py.File(h5_path, "r") as f:
            try:
                require_format(f["config"], str(h5_path))
            except ValueError as e:
                self._bad[str(h5_path)] = str(e)
                return np.zeros(0, np.int64)
            ident = read_ident(f)
        self._ident_by_path[str(h5_path)] = ident
        ev = _identity_ids(ident["source_file"], ident["event"], run=ident["run"])
        self._id2pos[str(h5_path)] = {int(e): i for i, e in enumerate(ev)}
        return np.asarray(ev, np.int64)

    def read_event(self, idx):
        """One event's record (:data:`helix.data.cells.SCHEMA` names) plus ``_ident``."""
        from helix.data.cells import read_event
        self._ensure_open()
        file_idx, event_id = self.locate(idx)
        path = str(self.h5_files[file_idx])
        pos = self._id2pos[path][int(event_id)]
        ident = self._ident_by_path[path]
        out = read_event(self._h5data[file_idx], pos)
        out["_ident"] = dict(run=ident["run"][pos], source_file=ident["source_file"][pos],
                             event=int(ident["event"][pos]))
        return out

def _as_str_list(arr, n):
    """h5py string column -> list[str] (``None`` -> empty strings)."""
    if arr is None:
        return [""] * n
    return [s.decode() if isinstance(s, bytes) else str(s) for s in arr]


def _identity_ids(source_file, event, run=None):
    """Per-event identity for the joint index: ``(run, source_file, event)``.

    ``/ident/event`` is the event's index WITHIN its source file, so it restarts
    at 0 in every file. A shard that spans more than one source file — which is
    exactly what loader mode produces, since it reads a run-wide joint index that
    crosses file boundaries — therefore contains repeated ids, and a map keyed on
    the id alone silently collapses them (measured: 250 rows into 200 slots, so
    20% of a 1000-event corpus was unreachable).

    ``run`` is load-bearing too: every detector run restarts shard numbering at
    ``*_sensor_0000.h5``, so ``(source_file, event)`` alone collides across runs.
    That is the identity the design doc specifies, and it is what a multi-run
    corpus needs. Encoded as ``file_ordinal << 32 | event`` for the common
    single-run case (keeps ids ORDERED, so iteration still follows the data), and
    as a stable 64-bit hash once more than one run is present or the source names
    are not numbered.
    """
    ev = np.asarray(event, np.int64)
    if source_file is None:
        return ev
    names = [s.decode() if isinstance(s, bytes) else str(s) for s in source_file]
    runs = ([r.decode() if isinstance(r, bytes) else str(r) for r in run]
            if run is not None else [""] * len(names))

    # Layout: [run 22b | file_ordinal 16b | event 24b] = 62 bits, positive int64.
    # The run goes in the HIGH bits so ordering WITHIN a run still follows the
    # data (index 0 is the first event, which a full hash would scramble), while
    # different runs occupy disjoint ranges. That matters because _index_for_shard
    # sees one shard — hence one run — at a time, so it cannot detect a cross-run
    # collision itself: every run restarts its shard numbering at *_sensor_0000.h5.
    _EVB, _FILEB = 24, 16
    sufs, ok = [], True
    for n in names:
        stem = n[:-3] if n.endswith(".h5") else n
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit() and int(tail) < (1 << _FILEB):
            sufs.append(int(tail))
        else:
            ok = False
            break
    if ok and ev.size and int(ev.max()) < (1 << _EVB) and int(ev.min()) >= 0:
        rh = np.array(
            [0 if not r else
             int.from_bytes(hashlib.blake2b(r.encode(), digest_size=8).digest(),
                            "little") & 0x3FFFFF
             for r in runs], np.int64)
        return (rh << np.int64(_EVB + _FILEB)) | \
               (np.asarray(sufs, np.int64) << np.int64(_EVB)) | ev
    # names not numbered, or counts beyond the field widths: fall back to a full
    # hash. Ordering is lost, which is cosmetic; a collision deletes events.
    return np.array(
        [int.from_bytes(
            hashlib.blake2b(f"{r}#{n}#{int(e)}".encode(), digest_size=8).digest(),
            "little") & 0x7FFFFFFFFFFFFFFF
         for r, n, e in zip(runs, names, ev.tolist())], np.int64)
