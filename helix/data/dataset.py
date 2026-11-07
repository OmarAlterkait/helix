"""The coeff corpus as a pimm-data family: the shard reader and ``CoeffTPCDataset`` (the FM's
input).

READER
======
The flat-columnar wavelet-coefficient shards (:mod:`helix.core.coeff_io` writes them).
The coeff corpus is a purpose-built training cache (produced by helix DSP), read
hot in the loop, so it is **flat columnar**, not per-event groups: shard-wide
``/coord`` + ``/value`` arrays plus an ``event_offset`` index; one contiguous
slice per event; ``plane_gid`` is a column. This is pimm-data's own offset idiom
(cf. the optical second row-space) applied shard-wide.

One modality per file (``{name}_coeff_*.h5`` noisy input, ``{name}_coeff_clean_*``
the noise-free values of a ``--keep-clean`` build), joined by identity — never mixed. The on-disk layout is the
shared contract of ``docs/ARCHITECTURE.md`` §6 (Coefficient shards); this reader
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

DATASET
=======
A single-family sibling of :class:`~pimm_data.optical.OpticalDataset`. It composes
:class:`CoeffTPCReader` (above) over flat-columnar coeff
shards and emits a **nested** dict with a ``coeff`` part of per-coefficient rows::

    {
      'coeff': {
        'band':      (M,)   int32    — band index [cA, cD_L, …, cD_1]
        'plane_gid': (M,)   int32    — canonical plane id (v*3 + {U,V,Y})
        'wire':      (M,)   int32    — signal index within the plane
        'tau':       (M,)   int32    — within-band coeff index
        'value':     (M, 1) float32  — RAW coeff (normalized at tokenize)
      },
      'name': str, 'split': str,
    }

Other modalities join by identity: ``'coeff_clean'`` (``modalities=('coeff','coeff_clean')``)
is the noise-free values of a ``--keep-clean`` build, its own shard family, read by no
training config.
``'cells'`` (with ``'coeff'``) adds ``sample['cells']``: the simulation truth on the
fine grid, one record per event (:mod:`helix.data.cells`), for downstream and
evaluation configs only -- pretraining never opens it.
This dataset only yields ROWS — tokenization belongs to the model side and lives
in ``helix.model.tokenize`` (:class:`helix.model.tokenize.CoeffTokenize`), which a recipe
registers into :data:`pimm_data.TRANSFORMS`. The shard tables the tokenizer needs
(``gids``/``n_wires``/``band_lengths``/``norm_sigma``) ride along in
``sample['coeff']['_meta']`` so a DataLoader worker is self-sufficient.

Canonical Collect (coeff rows = one part, offset counts coeffs/event)::

    CoeffTPCDataset(data_root=..., dataset_name='...', transform=[
        dict(type='Collect', parts={'coeff': dict(
            keys=('band', 'plane_gid', 'wire', 'tau', 'value'), feat_keys=('value',),
            offset_keys_dict=dict(offset='band'))})])

Registered in :data:`pimm_data.DATASETS`.
"""

import glob
import hashlib
import logging
import os

import numpy as np
import h5py

from pimm_data import ShardEventDataset, ShardReaderBase
from pimm_data.builder import DATASETS

from helix.core.coeff_io import FORMAT, coord_digest, shard_format

logger = logging.getLogger(__name__)


# ---- reader ----------------------------------------------------------------------------------

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
            raise ValueError(f"{path}: format {fmt!r}, not {FORMAT!r}; rebuild it "
                             f"(scripts/migrate_format.py at tag archive-2026-11 migrated the format-1 corpora)")
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
    # names not numbered (a campaign task's sensor_<task>.h5, SBND's time-stamped
    # files): [hash(run, file) 39b | event 24b], so events still follow the data
    # within a file; events beyond the field width: a full hash (ordering lost,
    # which is cosmetic; a collision deletes events).
    if ev.size and int(ev.max()) < (1 << _EVB) and int(ev.min()) >= 0:
        fh = np.array([int.from_bytes(hashlib.blake2b(f"{r}#{n}".encode(), digest_size=8).digest(),
                                      "little") & ((1 << 39) - 1) for r, n in zip(runs, names)], np.int64)
        return (fh << np.int64(_EVB)) | ev
    return np.array(
        [int.from_bytes(
            hashlib.blake2b(f"{r}#{n}#{int(e)}".encode(), digest_size=8).digest(),
            "little") & 0x7FFFFFFFFFFFFFFF
         for r, n, e in zip(runs, names, ev.tolist())], np.int64)


# ---- dataset ---------------------------------------------------------------------------------

@DATASETS.register_module()
class CoeffTPCDataset(ShardEventDataset):
    """Flat-columnar wavelet-coefficient corpus (noisy ``coeff`` + optional
    joined ``coeff_clean`` target)."""

    VALID_MODALITIES = ('coeff', 'coeff_clean', 'cells')

    def __init__(
        self,
        data_root,
        split='',
        modalities=('coeff',),
        dataset_name='coeff_tpc',
        transform=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        strict_lengths=True,
        event_range=None,
        exclude_range=None,
        holdout=None,
        split_role=None,
        max_event_size=None,
        n_events=None,
        subset_seed=0,
    ):
        # strict by DEFAULT, unlike the other datasets. A coeff corpus is built
        # by hundreds of independent jobs, and four separate failure shapes —
        # a missing coeff_clean pair, a truncated shard, a shard absent mid-run,
        # a job that died after writing only its noisy file — all resolve to the
        # joint index quietly ALIGNING on what it found and emitting a log
        # warning. Each silently shortens the corpus (measured: 25% gone from a
        # missing pair) in a way no downstream check can see. Pass
        # strict_lengths=False deliberately to read a partial corpus mid-build.
        self._modalities = tuple(modalities)
        self._validate_modalities(self._modalities)
        self._dataset_name = dataset_name
        self._max_len = max_len
        self._strict_lengths = strict_lengths
        self._source_data_root = data_root
        self._source_split = split

        self.coeff_reader = None
        self.coeff_clean_reader = None
        self.cells_reader = None
        if 'coeff' in self._modalities:
            self.coeff_reader = CoeffTPCReader(
                data_root=self._modality_root('coeff'), split=split,
                dataset_name=dataset_name, modality='coeff')
        if 'coeff_clean' in self._modalities:
            self.coeff_clean_reader = CoeffTPCReader(
                data_root=self._modality_root('coeff_clean'), split=split,
                dataset_name=dataset_name, modality='coeff_clean')
        if 'cells' in self._modalities:
            if 'coeff' not in self._modalities:
                raise ValueError("modalities: 'cells' is checked against the coeff event's identity; "
                                 "load it with 'coeff'")
            self.cells_reader = CellsReader(
                data_root=self._modality_root('cells'), split=split, dataset_name=dataset_name)

        # consumed by ShardEventDataset.get_data_list
        self._event_range = tuple(event_range) if event_range else None
        self._exclude_range = tuple(exclude_range) if exclude_range else None
        self._holdout = dict(holdout) if holdout else None
        self._split_role = split_role
        self._canonical_reader = self.coeff_reader or self.coeff_clean_reader
        self._build_joint_index(source_label=f"CoeffTPCDataset({data_root!r})")
        self._max_event_size = max_event_size

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, ignore_index=ignore_index, loop=loop,
        )
        # AFTER super().__init__, because `data_list` is what the base class
        # builds there. Calling it before gets an AttributeError, and calling it
        # before the joint index exists would filter against the wrong table.
        self._drop_oversize_events()
        if n_events is not None:
            self._subset(int(n_events), int(subset_seed))

    def _subset(self, n, seed):
        """Keep a fixed random ``n`` of this split's events, in corpus order.

        The label-efficiency control: a downstream task trained on 64, 256, ... labelled
        events. The draw is a prefix of one permutation of the split (``seed``), so the subsets
        of one seed are NESTED -- a larger arm holds every event of a smaller one -- and
        independent of the GPU count and the loader.
        """
        if n > len(self.data_list):
            raise ValueError(f"n_events={n}, but the split holds {len(self.data_list)} events")
        pick = np.sort(np.random.default_rng(seed).permutation(len(self.data_list))[:n])
        self.data_list = [self.data_list[i] for i in pick]

    def _drop_oversize_events(self):
        """Remove events too large to fit in GPU memory.

        The model takes ONE event per rank, so a step costs the largest event in
        it, and peak memory grows with the coefficient count -- the categorical
        head alone is ``(n_cells, n_slot, K)``. Measured on a 40 GB A100: the
        five largest events in a run (697k-725k coefficients) all run out of
        memory in a single forward/backward, while the median event (271k) peaks
        around 13 GiB.

        This only bites at SCALE, which is why it can sit unnoticed. Each step
        draws one event per rank, so the largest in a step is the p_(1-1/N)
        quantile: at 16 ranks the top 0.1% appears every few hundred steps, at
        512 ranks it appears in roughly every other step. And a single rank
        running out of memory does not fail fast -- it leaves every other rank
        blocked in a collective until the NCCL watchdog fires.

        Over the eight-run corpus a 600,000-coefficient cap drops 183 events of
        157,991: **0.12% of events and 0.27% of coefficients**. That is the whole
        cost of removing the failure mode.

        ``None`` disables it, which is right for single-GPU work where the tail
        is harmless and dropping data would be gratuitous.
        """
        if not self._max_event_size:
            return
        r = self._canonical_reader
        if r is None or not hasattr(r, "event_sizes"):
            return
        import numpy as _np
        sizes = _np.asarray(r.event_sizes(), dtype=_np.int64)
        keep = [i for i in self.data_list if sizes[i] <= self._max_event_size]
        dropped = len(self.data_list) - len(keep)
        if dropped:
            logger.info(
                "CoeffTPCDataset: dropped %d of %d events over %d coefficients "
                "(%.2f%%) -- they do not fit in GPU memory one-per-rank",
                dropped, len(self.data_list), self._max_event_size,
                100.0 * dropped / max(1, len(self.data_list)))
        self.data_list = keep

    def event_identity(self, idx):
        """``(run, source_file, event)`` — the SIMULATION event, hashed to ints.

        Stronger than the base's ``(config_id, file_index, event_id)``: those are
        properties of this corpus's sharding, while ``/ident`` names the event in
        the upstream simulation. So a split keyed on this survives not only shard
        add/remove/reorder but a full corpus REBUILD with a different wavelet,
        noise model or shard size — which matters because the probe holdout must
        keep meaning the same events across exactly that kind of change.
        """
        import hashlib
        t = self._ident_at(idx)
        h = hashlib.blake2b(f"{t['run']}/{t['source_file']}".encode(),
                            digest_size=8).digest()
        return (int.from_bytes(h, 'big') >> 1, t['event'])   # >>1 keeps it positive

    def _ident_at(self, idx):
        """``{run, source_file, event}`` for a GLOBAL index, without reading it."""
        r = self._canonical_reader
        file_idx, event_num = r.locate(idx)
        path = str(r.h5_files[file_idx])
        ident = r._ident_by_path[path]
        pos = r._id2pos[path][int(event_num)]
        return dict(run=ident['run'][pos], source_file=ident['source_file'][pos],
                    event=int(ident['event'][pos]))

    def holdout_manifest(self):
        """``[{run, source_file, event}, ...]`` for this split, in index order.

        The written counterpart to the hash: reproducible is not the same as
        auditable, and a probe holdout that exists only as code cannot be
        inspected, diffed, or cited.
        """
        r = self._canonical_reader
        out = []
        for i in range(len(self.data_list)):
            # Straight from the reader's index — NOT get_data(), which loads the
            # event's full coefficient arrays. Resolving 20k identities that way
            # meant 20k complete event reads for data that is already in memory.
            out.append(self._ident_at(self.data_list[i]))
        return out

    def event_sizes(self):
        """Per-event coefficient counts, aligned with THIS dataset's index.

        The reader's table is over its own global index; ``data_list`` is this
        split's selection into it (holdout, event_range, max_len), so the two are
        only the same while data_list is ``range(n)``. Mapping through it is what
        keeps a size table usable after a split narrows the corpus -- the same
        distinction ``get_data`` makes one line below.
        """
        r = self._canonical_reader
        if r is None or not hasattr(r, "event_sizes"):
            return None
        import numpy as _np
        sizes = _np.asarray(r.event_sizes(), dtype=_np.int64)
        return sizes[_np.asarray(self.data_list, dtype=_np.int64)]

    def get_data(self, idx):
        # data_list VALUES are event ids. Identical to `idx % len(...)` while
        # the list is range(n), and the only correct form once it is not.
        real_idx = self.data_list[idx % len(self.data_list)]
        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }
        raw_noisy = None
        if self.coeff_reader is not None:
            raw_noisy = self.coeff_reader.read_event(real_idx)
            data['coeff'] = self._build_coeff(raw_noisy, meta=self._shard_meta())
            # The source event this coeff event was built from: (run, source_file,
            # event). Carried at the TOP level rather than inside a part, because
            # it identifies the sample, not a row-space — and because a consumer
            # joining truth onto it must not have to guess which part to look in.
            # This is what lets a probe reach the simulation truth without the
            # corpus storing it: /ident is already written by every shard.
            if '_ident' in raw_noisy:
                data['ident'] = raw_noisy['_ident']
        if self.coeff_clean_reader is not None:
            raw_clean = self.coeff_clean_reader.read_event(real_idx)
            if 'band' not in raw_clean:                  # values-only target
                raw_clean = self._inherit_coords(raw_clean, raw_noisy, real_idx)
            # a clean-only dataset still needs the shard tables for a tokenizer
            data['coeff_clean'] = self._build_coeff(
                raw_clean, meta=None if raw_noisy is not None else self._shard_meta())
        if self.cells_reader is not None:
            cells = self.cells_reader.read_event(real_idx)
            ident = cells.pop('_ident')
            if ident != raw_noisy.get('_ident'):
                raise ValueError(f"event {real_idx}: cells are for {ident}, the coeff event is "
                                 f"{raw_noisy.get('_ident')} -- MISPAIRED coeff/cells shards.")
            data['cells'] = cells
        return data

    def _inherit_coords(self, raw_clean, raw_noisy, idx):
        """Supply a values-only target's coords from its paired noisy event.

        The clean target is CO-SUPPORTED, so its coords are the noisy event's and
        are not stored twice (13 of every 34 bytes in a pair). ``coord_digest``
        is what makes that safe: without the check a mispaired shard misaligns
        every target row silently, and the model trains against noise.
        """
        if raw_noisy is None:
            raise ValueError(
                "modalities=('coeff_clean',) cannot be read alone: the clean shard is "
                "values-only and inherits its coords from the paired 'coeff' shard. "
                "Use modalities=('coeff', 'coeff_clean').")
        if raw_clean['value'].shape[0] != raw_noisy['value'].shape[0]:
            raise ValueError(
                f"event {idx}: clean has {raw_clean['value'].shape[0]} rows, noisy has "
                f"{raw_noisy['value'].shape[0]} — the shards are not co-supported.")
        want, got = raw_clean['_coord_digest'], raw_noisy['_coord_digest']
        if int(got) != int(want):
            raise ValueError(
                f"event {idx}: coord_digest mismatch (clean {int(want):#018x} vs noisy "
                f"{int(got):#018x}) — MISPAIRED coeff/coeff_clean shards.")
        return {**raw_clean, **{k: raw_noisy[k]
                                for k in ('band', 'plane_gid', 'wire', 'tau')}}

    def _shard_meta(self):
        """Shard-level tables a tokenizer needs, travelling WITH the sample.

        A tokenizer runs in a DataLoader worker and cannot reach back to the
        dataset object, so the shard's ``/config`` tables ride along. They are
        small (a few hundred bytes) and ``Collect`` drops ``_meta``, so they never
        reach the batch. Note ``norm_sigma`` rows are ordered by POSITION in
        ``gids`` — consumers must resolve through ``gids``, never index by gid.
        """
        r = self._canonical_reader
        return dict(gids=r.gids, n_wires=r.n_wires,
                    band_lengths=r.band_lengths, norm_sigma=r.norm_sigma,
                    sigma_norm=r.sigma_norm)

    @staticmethod
    def _build_coeff(raw, meta=None):
        """Neutral coeff rows: coords as columns + value as an (M,1) feature. A shard
        with per-wire validity adds ``wire_valid`` (G, max n_wires) to ``_meta``."""
        if meta and raw.get('_wire_valid') is not None:
            from helix.core.coeff_io import unpack_wire_valid
            meta = {**meta, 'wire_valid': unpack_wire_valid(raw['_wire_valid'], meta['n_wires'])}
        return {
            'band': raw['band'],
            'plane_gid': raw['plane_gid'],
            'wire': raw['wire'],
            'tau': raw['tau'],
            'value': raw['value'][:, None].astype(np.float32),
            **({'_meta': meta} if meta else {}),
        }

    @property
    def norm_sigma(self):
        """The shard's cross-event normalization σ table (n_gid, n_bands), for the
        tokenizer. Rows are indexed by POSITION in :attr:`gids`, not gid value — a
        tokenizer must map ``gids.index(gid)`` to the row (gids may be non-contiguous)."""
        return self._canonical_reader.norm_sigma if self._canonical_reader else None

    @property
    def gids(self):
        """Plane gids (the row order of :attr:`norm_sigma` / sigma_threshold)."""
        return self._canonical_reader.gids if self._canonical_reader else None

    @property
    def band_lengths(self):
        return self._canonical_reader.band_lengths if self._canonical_reader else None
