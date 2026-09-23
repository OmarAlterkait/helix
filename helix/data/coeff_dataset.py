"""CoeffTPCDataset — the wavelet-coefficient corpus (the FM's input).

A single-family sibling of :class:`~pimm_data.optical.OpticalDataset`. It composes
:class:`~pimm_data.readers.coeff_tpc.CoeffTPCReader` over flat-columnar coeff
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

Targets are separate joined modalities (``modalities=('coeff','coeff_clean')``):
the clean target is its own ``coeff_clean`` shard family, aligned by identity.
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

import numpy as np

from pimm_data.builder import DATASETS
from pimm_data import ShardEventDataset

from helix.core.coeff_io import coord_digest
import logging

from helix.data.coeff_reader import CoeffTPCReader

logger = logging.getLogger(__name__)


@DATASETS.register_module()
class CoeffTPCDataset(ShardEventDataset):
    """Flat-columnar wavelet-coefficient corpus (noisy ``coeff`` + optional
    joined ``coeff_clean`` target)."""

    VALID_MODALITIES = ('coeff', 'coeff_clean')

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
        if 'coeff' in self._modalities:
            self.coeff_reader = CoeffTPCReader(
                data_root=self._modality_root('coeff'), split=split,
                dataset_name=dataset_name, modality='coeff')
        if 'coeff_clean' in self._modalities:
            self.coeff_clean_reader = CoeffTPCReader(
                data_root=self._modality_root('coeff_clean'), split=split,
                dataset_name=dataset_name, modality='coeff_clean')

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
        if t is None:                           # legacy shard with no /ident
            file_idx, event_num = self._canonical_reader.locate(idx)
            return (0, file_idx, event_num)
        h = hashlib.blake2b(f"{t['run']}/{t['source_file']}".encode(),
                            digest_size=8).digest()
        return (int.from_bytes(h, 'big') >> 1, t['event'])   # >>1 keeps it positive

    def _ident_at(self, idx):
        """``{run, source_file, event}`` for a GLOBAL index, without reading it."""
        r = self._canonical_reader
        file_idx, event_num = r.locate(idx)
        path = str(r.h5_files[file_idx])
        ident = r._ident_by_path.get(path)
        if ident is None:
            return None
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
            t = self._ident_at(self.data_list[i])
            if t is not None:
                out.append(t)
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
        want = raw_clean.get('_coord_digest')
        if want is None:
            # values-only + no digest = unverifiable pairing. Narrow on purpose:
            # this fires only for has_coords=False shards, so pre-digest legacy
            # shards (which carry their own coords) keep reading.
            raise ValueError(
                f"event {idx}: the coeff_clean shard is values-only but carries no "
                f"coord_digest — its pairing with 'coeff' cannot be verified. "
                f"Rebuild it with a current writer.")
        got = raw_noisy.get('_coord_digest')
        if got is None:
            got = coord_digest(raw_noisy['band'], raw_noisy['plane_gid'],
                               raw_noisy['wire'], raw_noisy['tau'])
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
        """Neutral coeff rows: coords as columns + value as an (M,1) feature."""
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
