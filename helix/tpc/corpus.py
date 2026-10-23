"""Corpus builder — compose the TPC chain into coeff-corpus shards.

Turns detector events into the flat-columnar coeff corpus (COEFF_CORPUS_DESIGN.md):
per event, the noisy input goes through :class:`helix.tpc.chain.TPCChain` (simulation:
:func:`helix.tpc.chain.sim_config`), one TPC at a time, and becomes a CoeffEvent
(:func:`helix.tpc.chain_coeff.coeff_event`). The clean target is the SAME transform of
the noise-free image evaluated at the *noisy support* (co-supported, so noisy↔clean align
row-for-row — the old ``star_tpc.prep_tpc_rows`` ``val_clean = clean[gid][b][noisy_mask]``;
:func:`helix.tpc.chain_coeff.clean_coeff_event`). Values are stored RAW; the cross-event
normalization table (``norm_sigma`` = mean over cal events of the per-event threshold σ)
goes in ``/config``, applied at tokenize — the design's raw-values-plus-sidecar.

The noise source is injected as a ``plane_fn(event) -> (noisy_planes, clean_planes)``
so this module stays pure helix DSP (no pimm-data import); a real builder passes a
``plane_fn`` that reads sensor shards and injects noise. ``{gid: image}`` dicts key
planes by canonical gid (``tpc * 3 + plane``, :func:`helix.tpc.readout.canonical_plane_gid`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from helix.core.coeff_io import write_coeff_shard, audit_shard
from helix.tpc import chain_coeff


def process_planes(chain, noisy_planes):
    """``{gid: (wires, ticks) image}`` -> the chain's plane results, every TPC present.

    Planes are grouped by TPC (``gid // n_planes``) and each TPC goes through
    :meth:`~helix.tpc.chain.TPCChain.process_images` in plane order. A TPC must bring
    all of its planes, with the readout's wire count each."""
    n_planes = len(chain.readout.planes)
    by_tpc = {}
    for gid, img in noisy_planes.items():
        by_tpc.setdefault(int(gid) // n_planes, {})[int(gid) % n_planes] = img
    results = []
    for tpc in sorted(by_tpc):
        planes = by_tpc[tpc]
        if tpc >= chain.readout.n_tpc:
            raise ValueError(f"plane gids of tpc {tpc}, but the readout has {chain.readout.n_tpc} tpcs")
        if sorted(planes) != list(range(n_planes)):
            raise ValueError(f"tpc {tpc}: planes {sorted(planes)} present; the chain needs all {n_planes}")
        rows = chain.readout.rows(tpc)
        for p in range(n_planes):
            if int(planes[p].shape[0]) != len(rows[p]):
                raise ValueError(f"gid {tpc * n_planes + p}: image has {int(planes[p].shape[0])} wires, "
                                 f"the readout {len(rows[p])} — geometry must match")
        results += chain.process_images([planes[p] for p in range(n_planes)], tpc)
    return results


def normalization_table(events) -> np.ndarray:
    """``norm_sigma`` (n_gid, n_bands) = mean over ``events`` of per-event
    ``sigma_threshold`` (= mean ``median(|gated band|)/0.6745`` over cal events —
    the old ``sigma_tab``)."""
    tab = np.stack([e.sigma_threshold for e in events]).mean(axis=0).astype(np.float32)
    if tab.size and float(np.nanmax(tab)) <= 0.0:
        raise ValueError(
            "norm_sigma is all zero — the per-event threshold sigma never reached "
            "CoeffEvent.sigma_threshold, so the tokenizer's normalization table "
            "would be meaningless. (This exact loss happened once: a flat-bands "
            "branch of from_sparse_results skipped the sigma assignment.)")
    return tab


def build_corpus(events, plane_fn, chain, out_dir, *,
                 dataset_name="coeff_tpc", run="", file_index=0, global_event_offset=0,
                 cal_events=None, norm_sigma=None, with_clean=True, write=True,
                 noise=None, provenance=None):
    """Build coeff (+ coeff_clean) shards from ``events`` via ``plane_fn`` and ``chain``
    (a :class:`helix.tpc.chain.TPCChain`).

    ``plane_fn(event) -> (noisy_planes, clean_planes)`` with ``{gid: (nw, nt) image}``
    (clean_planes may be ``{}`` when ``with_clean=False``). Returns
    ``(noisy_events, clean_events, norm_sigma)``. Cal events index into ``events``.

    Normalization: pass ``norm_sigma`` to freeze ONE table across every shard of
    a corpus (the correct production setting). Otherwise ``cal_events=None``
    (default) averages EVERY event in this shard, which has no sampling error but
    still differs from other shards. The old 2-event window inherited from
    star_tpc is the worst option: an unlucky pair is 10% off the shard mean.
    """
    # ``events`` may be plain ints, or (source_file, event[, noise_seed]) tuples
    # carrying identity/provenance; plane_fn always receives just the event index.
    stream = ((ev,) + plane_fn(ev if not isinstance(ev, (tuple, list)) else ev[1])
              for ev in events)
    return build_corpus_stream(stream, chain, out_dir, dataset_name=dataset_name,
                               run=run, file_index=file_index,
                               global_event_offset=global_event_offset,
                               cal_events=cal_events, norm_sigma=norm_sigma,
                               with_clean=with_clean, write=write, noise=noise, provenance=provenance)


def _identity(ev, default_src):
    """Stream item id -> ``(source_file, event, noise_seed)``.

    An ``int`` means "event N of ``default_src``" (serial mode, which reads one
    named file). A ``(source_file, event)`` pair carries the TRUE origin, which
    is what a multi-shard loader must pass: it reads a joint index over many
    files, so a fabricated ``f"{name}_sensor_{file_index:04d}.h5"`` would record
    provenance that never matches the bytes. An optional third element is the
    RESOLVED noise seed — the corpus stores one fixed noise realisation per event
    by design, so the seed is the only thing that makes that realisation
    reproducible, and it is not otherwise derivable (the two build modes use
    different seed formulas).
    """
    if isinstance(ev, (tuple, list)):
        src, e = ev[0], ev[1]
        seed = int(ev[2]) if len(ev) > 2 and ev[2] is not None else -1
        return str(src), int(e), seed
    return default_src, int(ev), -1


def build_corpus_stream(stream, chain, out_dir, *,
                        dataset_name="coeff_tpc", run="", file_index=0,
                        global_event_offset=0, cal_events=None, norm_sigma=None,
                        with_clean=True, write=True, progress=None, noise=None, provenance=None):
    """Build shards from a STREAM of ``(event_id, noisy_planes, clean_planes)``.

    The stream form is what a DataLoader gives (sequential, prefetched in worker
    processes); :func:`build_corpus` is the random-access wrapper over it. Planes
    may be numpy arrays or tensors; ``chain`` moves them to its device.

    ``event_id`` is either an ``int`` (event N of this shard's own source file)
    or a ``(source_file, event)`` pair when the producer reads across files.
    """
    out_dir = Path(out_dir)
    noisy_ces, clean_ces, seeds = [], [], []
    default_src = f"{dataset_name}_sensor_{file_index:04d}.h5"
    for i, (ev, noisy_planes, clean_planes) in enumerate(stream):
        src, ev_id, seed = _identity(ev, default_src)
        seeds.append(seed)
        results = process_planes(chain, noisy_planes)
        ce = chain_coeff.coeff_event(chain, results, run=run, source_file=src, event=ev_id)
        noisy_ces.append(ce)
        if with_clean:
            clean_ces.append(chain_coeff.clean_coeff_event(chain, ce, clean_planes))
        if progress is not None:
            progress(i, ce)

    # --- normalization table ---------------------------------------------
    # Priority: an externally supplied GLOBAL table > all events > a subset.
    #
    # norm_sigma must be the SAME for every shard of a corpus. If each shard
    # derives its own, the identical physical coefficient is normalised
    # differently depending on which shard it landed in — measured shard-to-shard
    # disagreement is 0.28% median / 0.94% max at 50 events/shard, 0.78% / 2.7% at
    # 16 — and the model cannot tell that apart from real signal. So a production
    # build computes the table ONCE and passes it to every shard.
    if not noisy_ces:
        raise ValueError(
            "build_corpus_stream: the stream yielded no events — nothing to build. "
            "(In loader mode this usually means --event-start is past the end of "
            "the run's joint index.)")
    n_bands = noisy_ces[0].sigma_threshold.shape[1]
    if norm_sigma is not None:
        norm = np.asarray(norm_sigma, np.float32)
        want = (len(noisy_ces[0].gids), n_bands)
        if norm.shape != want:
            raise ValueError(f"norm_sigma shape {norm.shape} != expected {want}")
        if float(np.nanmax(norm)) <= 0.0:
            raise ValueError("supplied norm_sigma is all zero")
    elif cal_events is None:
        norm = normalization_table(noisy_ces)        # ALL events: no sampling error
    else:
        if max(cal_events) >= len(noisy_ces):
            raise ValueError(
                f"cal_events {tuple(cal_events)} out of range for {len(noisy_ces)} built "
                f"events; pass cal_events=None to use every event")
        norm = normalization_table([noisy_ces[i] for i in cal_events])

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        kw = dict(dataset_name=dataset_name, file_index=file_index,
                  global_event_offset=global_event_offset, norm_sigma=norm,
                  noise=noise, provenance=provenance,
                  noise_seeds=(seeds if any(s >= 0 for s in seeds) else None))
        p_noisy = out_dir / f"{dataset_name}_coeff_{file_index:04d}.h5"
        write_coeff_shard(p_noisy, noisy_ces, **kw)
        audit_shard(p_noisy)                 # refuse to ship a degenerate shard
        if with_clean:
            # values-only: the clean target is co-supported, so its coords are
            # the noisy shard's. coord_digest (in both) makes the join checkable.
            p_clean = out_dir / f"{dataset_name}_coeff_clean_{file_index:04d}.h5"
            write_coeff_shard(p_clean, clean_ces, coords=False, **kw)
            audit_shard(p_clean)
    return noisy_ces, clean_ces, norm
