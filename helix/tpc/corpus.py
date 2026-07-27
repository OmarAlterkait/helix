"""Corpus builder — compose the DSP into coeff-corpus shards.

Turns detector events into the flat-columnar coeff corpus (COEFF_CORPUS_DESIGN.md):
per event, the noisy input is `process_plane(removal='gate')`; the clean target is
the SAME wavelet coefficients evaluated at the *noisy support* (co-supported, so
noisy↔clean align row-for-row — exactly the old ``star_tpc.prep_tpc_rows`` where
``val_clean = clean[gid][b][noisy_mask]``). Values are stored RAW; the cross-event
normalization table (``norm_sigma`` = mean over cal events of the per-event
threshold σ = ``median(|gated band|)/0.6745``) goes in ``/config``, applied at
tokenize — the design's raw-values-plus-sidecar (vs the old baked-in ``SIGMA/σ``).

The noise source is injected as a ``plane_fn(event) -> (noisy_planes, clean_planes)``
so this module stays pure helix DSP (no pimm-data import); a real builder passes a
``plane_fn`` that reads sensor shards and injects noise via pimm-data (colored
spectrum). ``{gid: image}`` dicts key planes by canonical gid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from helix.core.wavelet import wavedec
from helix.core.coeff_event import CoeffEvent, _is_device, _xfer_cap
from helix.core.coeff_io import write_coeff_shard, audit_shard
from helix.tpc.config import DetectorConfig
from helix.tpc.pipeline import process_plane, event_coeff_event, _pad_time


def clean_coeff_event(ce_noisy: CoeffEvent, clean_planes: dict, config: DetectorConfig,
                      *, run: str = "", source_file: str = "", event: int = -1) -> CoeffEvent:
    """Clean-target CoeffEvent CO-SUPPORTED with ``ce_noisy``.

    The clean image is DWT'd (no removal, no threshold) and its coefficients are
    gathered at the noisy event's ``(gid, band, wire, tau)`` — identical coords,
    clean values (the old ``clean[gid][b][mask]``). ``sigma_threshold`` records the
    ``sigma_threshold`` is copied from the noisy event (see below).
    """
    clean_bands = {}
    for gid, img in clean_planes.items():
        xin = _pad_time(img, config.dwt_level)
        bands, _ = wavedec(xin, wavelet=config.wavelet, level=config.dwt_level,
                           mode=config.dwt_mode)
        # kept in backend-native form: on the jax backend both the per-band MAD
        # and the gather below run ON DEVICE (host np.median over the dense bands
        # costs ~363 ms/event, np.median on GPU ~16 ms).
        clean_bands[int(gid)] = bands

    # the clean planes must cover the noisy support with matching geometry — else the
    # gather at (gid, wire, tau) silently misaligns or crashes.
    for gi, gid in enumerate(ce_noisy.gids):
        g = int(gid)
        if g not in clean_bands:
            raise ValueError(f"clean_planes missing gid {g} present in the noisy event")
        if clean_bands[g][0].shape[0] != int(ce_noisy.n_wires[gi]):
            raise ValueError(
                f"gid {g}: clean has {clean_bands[g][0].shape[0]} wires, "
                f"noisy has {int(ce_noisy.n_wires[gi])} — geometry must match")

    values = np.zeros(ce_noisy.n_coeff, np.float32)
    # The clean target carries the NOISY event's sigma_threshold, not its own.
    # A noise-free image's DWT bands are >50% exact zeros, so their MAD sigma is
    # identically 0 — carrying that would write an all-zero field indistinguishable
    # from the bug where sigma was silently lost, while conveying nothing. The
    # sigma that actually applies to this target is the one its normalisation uses,
    # which is shared with the noisy input.
    sigma = ce_noisy.sigma_threshold.copy()
    n_bands = ce_noisy.basis.n_bands
    device = _is_device(next(iter(clean_bands.values()))[0])
    xp = np
    if device:
        import jax.numpy as jnp
        xp = jnp
    bl = np.asarray(ce_noisy.basis.band_lengths, np.int64)
    col_off = np.concatenate([[0], np.cumsum(bl)])[:-1]      # band -> column offset
    for gi, gid in enumerate(ce_noisy.gids):
        gid = int(gid)
        gmask = ce_noisy.plane_gid == gid
        bands_g = clean_bands[gid]
        if not gmask.any():
            continue
        # Gather ON DEVICE with PADDED indices. Two constraints collide here: a
        # device gather on a per-event index length retraces, but transferring the
        # dense band to gather on the host moves ~205 MB/event. Padding the index
        # arrays to a monotonic static cap satisfies both — one shape, and only
        # cap values come back instead of the whole band.
        cols = col_off[ce_noisy.band[gmask]] + ce_noisy.tau[gmask]
        rows = ce_noisy.wire[gmask]
        nsel = rows.shape[0]
        if device:
            cap = _xfer_cap(nsel)
            if cap > nsel:
                rows = np.concatenate([rows, np.zeros(cap - nsel, rows.dtype)])
                cols = np.concatenate([cols, np.zeros(cap - nsel, cols.dtype)])
            cat = bands_g.flat if hasattr(bands_g, "flat") else \
                xp.concatenate(list(bands_g), axis=1)      # already flat: no copy
            gv = cat[xp.asarray(rows), xp.asarray(cols)]
            values[gmask] = np.asarray(gv, np.float32)[:nsel]
        else:
            cat = np.concatenate(list(bands_g), axis=1)
            values[gmask] = cat[rows, cols]
    return CoeffEvent(
        band=ce_noisy.band.copy(), plane_gid=ce_noisy.plane_gid.copy(),
        wire=ce_noisy.wire.copy(), tau=ce_noisy.tau.copy(), value=values,
        gids=ce_noisy.gids.copy(), n_wires=ce_noisy.n_wires.copy(),
        sigma_threshold=sigma, basis=ce_noisy.basis,
        run=run or ce_noisy.run, source_file=source_file or ce_noisy.source_file,
        event=event if event >= 0 else ce_noisy.event)


def normalization_table(events) -> np.ndarray:
    """``norm_sigma`` (n_gid, n_bands) = mean over ``events`` of per-event
    ``sigma_threshold`` (= mean ``median(|gated band|)/0.6745`` over cal events —
    the old ``sigma_tab``)."""
    tab = np.stack([e.sigma_threshold for e in events]).mean(axis=0).astype(np.float32)
    if tab.size and float(np.nanmax(tab)) <= 0.0:
        raise ValueError(
            "norm_sigma is all zero — the per-event threshold sigma never reached "
            "CoeffEvent.sigma_threshold, so the tokenizer's normalization table "
            "would be meaningless. (This exact loss happened once: the flat/jax "
            "branch of from_sparse_results skipped the sigma assignment.)")
    return tab


def build_corpus(events, plane_fn, config: DetectorConfig, out_dir, *,
                 dataset_name="coeff_tpc", run="", file_index=0, global_event_offset=0,
                 cal_events=None, norm_sigma=None, with_clean=True, write=True):
    """Build coeff (+ coeff_clean) shards from ``events`` via ``plane_fn``.

    ``plane_fn(event) -> (noisy_planes, clean_planes)`` with ``{gid: (nw, nt) image}``
    (clean_planes may be ``{}`` when ``with_clean=False``). Returns
    ``(noisy_events, clean_events, norm_sigma)``. Cal events index into ``events``.

    Normalization: pass ``norm_sigma`` to freeze ONE table across every shard of
    a corpus (the correct production setting). Otherwise ``cal_events=None``
    (default) averages EVERY event in this shard, which has no sampling error but
    still differs from other shards. The old 2-event window inherited from
    star_tpc is the worst option: an unlucky pair is 10% off the shard mean.
    """
    stream = ((ev,) + plane_fn(ev) for ev in events)
    return build_corpus_stream(stream, config, out_dir, dataset_name=dataset_name,
                               run=run, file_index=file_index,
                               global_event_offset=global_event_offset,
                               cal_events=cal_events, norm_sigma=norm_sigma,
                               with_clean=with_clean, write=write)


def build_corpus_stream(stream, config: DetectorConfig, out_dir, *,
                        dataset_name="coeff_tpc", run="", file_index=0,
                        global_event_offset=0, cal_events=None, norm_sigma=None,
                        with_clean=True, write=True, progress=None):
    """Build shards from a STREAM of ``(event_id, noisy_planes, clean_planes)``.

    The stream form is what a DataLoader gives (sequential, prefetched in worker
    processes); :func:`build_corpus` is the random-access wrapper over it. Planes
    may be numpy or device arrays — ``process_plane`` dispatches on the backend.
    """
    out_dir = Path(out_dir)
    noisy_ces, clean_ces = [], []
    src = f"{dataset_name}_sensor_{file_index:04d}.h5"
    for i, (ev, noisy_planes, clean_planes) in enumerate(stream):
        results = {int(gid): process_plane(img, config, removal="gate", with_images=False)
                   for gid, img in noisy_planes.items()}
        ce = event_coeff_event(results, config, run=run, source_file=src, event=int(ev))
        noisy_ces.append(ce)
        if with_clean:
            clean_ces.append(clean_coeff_event(ce, clean_planes, config,
                                               run=run, source_file=src, event=int(ev)))
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
    n_bands = config_n_bands = noisy_ces[0].sigma_threshold.shape[1]
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
                  global_event_offset=global_event_offset, norm_sigma=norm)
        p_noisy = out_dir / f"{dataset_name}_coeff_{file_index:04d}.h5"
        write_coeff_shard(p_noisy, noisy_ces, **kw)
        audit_shard(p_noisy)                 # refuse to ship a degenerate shard
        if with_clean:
            p_clean = out_dir / f"{dataset_name}_coeff_clean_{file_index:04d}.h5"
            write_coeff_shard(p_clean, clean_ces, **kw)
            audit_shard(p_clean)
    return noisy_ces, clean_ces, norm
