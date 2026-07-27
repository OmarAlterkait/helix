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
from helix.core.coeff_event import CoeffEvent
from helix.core.coeff_io import write_coeff_shard
from helix.tpc.config import DetectorConfig
from helix.tpc.pipeline import process_plane, event_coeff_event, _pad_time


def clean_coeff_event(ce_noisy: CoeffEvent, clean_planes: dict, config: DetectorConfig,
                      *, run: str = "", source_file: str = "", event: int = -1) -> CoeffEvent:
    """Clean-target CoeffEvent CO-SUPPORTED with ``ce_noisy``.

    The clean image is DWT'd (no removal, no threshold) and its coefficients are
    gathered at the noisy event's ``(gid, band, wire, tau)`` — identical coords,
    clean values (the old ``clean[gid][b][mask]``). ``sigma_threshold`` records the
    clean bands' MAD (informational; the target is normalized with the shared
    ``norm_sigma`` at tokenize).
    """
    clean_bands = {}
    for gid, img in clean_planes.items():
        xin = _pad_time(img, config.dwt_level)
        bands, _ = wavedec(xin, wavelet=config.wavelet, level=config.dwt_level,
                           mode=config.dwt_mode)
        # host-materialise once: the gather below is numpy fancy-indexing, which
        # would otherwise re-cross PCIe per (gid, band) on the jax backend
        clean_bands[int(gid)] = [np.asarray(c) for c in bands]

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
    sigma = np.zeros_like(ce_noisy.sigma_threshold)
    n_bands = ce_noisy.basis.n_bands
    for gi, gid in enumerate(ce_noisy.gids):
        gid = int(gid)
        gmask = ce_noisy.plane_gid == gid
        for b in range(n_bands):
            cb = clean_bands[gid][b]
            sigma[gi, b] = np.median(np.abs(cb)) / 0.6745
            m = gmask & (ce_noisy.band == b)
            if m.any():
                values[m] = cb[ce_noisy.wire[m], ce_noisy.tau[m]]
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
    return np.stack([e.sigma_threshold for e in events]).mean(axis=0).astype(np.float32)


def build_corpus(events, plane_fn, config: DetectorConfig, out_dir, *,
                 dataset_name="coeff_tpc", run="", file_index=0, global_event_offset=0,
                 cal_events=(0, 1), with_clean=True, write=True):
    """Build coeff (+ coeff_clean) shards from ``events`` via ``plane_fn``.

    ``plane_fn(event) -> (noisy_planes, clean_planes)`` with ``{gid: (nw, nt) image}``
    (clean_planes may be ``{}`` when ``with_clean=False``). Returns
    ``(noisy_events, clean_events, norm_sigma)``. Cal events index into ``events``.
    """
    out_dir = Path(out_dir)
    noisy_ces, clean_ces = [], []
    src = f"{dataset_name}_sensor_{file_index:04d}.h5"
    for ev in events:
        noisy_planes, clean_planes = plane_fn(ev)
        results = {int(gid): process_plane(img, config, removal="gate", with_images=False)
                   for gid, img in noisy_planes.items()}
        ce = event_coeff_event(results, config, run=run, source_file=src, event=int(ev))
        noisy_ces.append(ce)
        if with_clean:
            clean_ces.append(clean_coeff_event(ce, clean_planes, config,
                                               run=run, source_file=src, event=int(ev)))

    if cal_events:
        if max(cal_events) >= len(noisy_ces):
            raise ValueError(
                f"cal_events {tuple(cal_events)} out of range for {len(noisy_ces)} built "
                f"events; pass cal_events=range(min(2, n)) or () to skip normalization")
        norm = normalization_table([noisy_ces[i] for i in cal_events])
    else:
        norm = None

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        kw = dict(dataset_name=dataset_name, file_index=file_index,
                  global_event_offset=global_event_offset, norm_sigma=norm)
        write_coeff_shard(out_dir / f"{dataset_name}_coeff_{file_index:04d}.h5", noisy_ces, **kw)
        if with_clean:
            write_coeff_shard(out_dir / f"{dataset_name}_coeff_clean_{file_index:04d}.h5",
                              clean_ces, **kw)
    return noisy_ces, clean_ces, norm
