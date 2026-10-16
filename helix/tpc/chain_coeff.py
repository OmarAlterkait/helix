"""The TPC chain's plane results as HELIX's corpus atom (:class:`helix.core.coeff_event.CoeffEvent`),
for simulation and data alike, written by :func:`helix.core.coeff_io.write_coeff_shard`.

``wire`` is the PHYSICAL wire number (the readout's), so a plane's uninstrumented wires are gaps
and cells keep their geometry; ``n_wires`` = the plane's physical wire count; ``wire_valid`` is
False for every wire without a valid row in the event (not instrumented, dead, noisy, baseline
step): the tokenizer gives those slots no target. ``sigma_threshold`` = the threshold sigma per
band (0 for dropped finest bands). The basis records the chain's coherent removal (with the
readout) and threshold, so ``basis_digest`` separates every configuration.
"""
from __future__ import annotations

import numpy as np

from helix.core.coeff_event import CoeffEvent
from helix.core.provenance import BasisDescriptor, padded_length_of
from helix.tpc.readout import canonical_plane_gid


def _host(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _nonzero(c):
    """(row, col, value) of a band's nonzeros, reduced on its device."""
    if hasattr(c, "detach"):
        idx = c.nonzero(as_tuple=False)
        r, t = idx[:, 0], idx[:, 1]
        return r.cpu().numpy(), t.cpu().numpy(), c[r, t].cpu().numpy()
    r, t = np.nonzero(c)
    return r, t, c[r, t]


def basis(chain, band_lengths, n_ticks, sigma_norm=2.6) -> BasisDescriptor:
    cfg = chain.cfg; d = cfg["dwt"]
    bl = tuple(int(x) for x in band_lengths)
    padded = padded_length_of(bl, d["wavelet"], d["level"], d["mode"])
    return BasisDescriptor(wavelet=d["wavelet"], level=int(d["level"]), mode=d["mode"], n_ticks_raw=int(n_ticks),
                           pad=int(padded - n_ticks), band_lengths=bl,
                           removal=dict(cfg["coherent"], readout=cfg["readout"], status=cfg["status"]["method"]),
                           threshold=dict(cfg["threshold"]), sigma_norm=float(sigma_norm))


def plane_gid(chain, result):
    return canonical_plane_gid(result["tpc"], chain.readout.planes.index(result["plane"]), len(chain.readout.planes))


def coeff_event(chain, results, *, run="", source_file="", event=-1, sigma_norm=2.6) -> CoeffEvent:
    """One event's plane results (:meth:`helix.tpc.chain.TPCChain.process_event` /
    ``process_images`` over every TPC) -> CoeffEvent."""
    ro = chain.readout
    by_gid = {plane_gid(chain, r): r for r in results}
    gids = np.array(sorted(by_gid), np.int32)
    n_wires = np.array([int(ro.wire[(ro.tpc == by_gid[g]["tpc"]) & (ro.plane == ro.planes.index(by_gid[g]["plane"]))].max()) + 1
                        for g in gids], np.int32)
    wire_valid = np.zeros((len(gids), int(n_wires.max())), bool)
    cols = {k: [] for k in ("band", "plane_gid", "wire", "tau", "value")}
    for gi, g in enumerate(gids):
        r = by_gid[int(g)]
        w_phys = ro.wire[np.asarray(r["rows"])]
        wire_valid[gi, w_phys[_host(r["valid"]).astype(bool)]] = True
        for b, c in enumerate(r["bands"]):
            row, t, v = _nonzero(c)
            cols["band"].append(np.full(len(row), b, np.uint8)); cols["plane_gid"].append(np.full(len(row), g, np.int32))
            cols["wire"].append(w_phys[row].astype(np.int32)); cols["tau"].append(t.astype(np.int32))
            cols["value"].append(np.asarray(v, np.float32))
    r0 = by_gid[int(gids[0])]
    bs = basis(chain, r0["band_lengths"], r0["n_ticks"], sigma_norm)
    return CoeffEvent(**{k: np.concatenate(v) for k, v in cols.items()}, gids=gids, n_wires=n_wires,
                      sigma_threshold=np.stack([_host(by_gid[int(g)]["sigma_per_band"]) for g in gids]).astype(np.float32),
                      basis=bs, run=str(run), source_file=str(source_file), event=int(event),
                      wire_valid=None if all(wire_valid[i, :n].all() for i, n in enumerate(n_wires)) else wire_valid)


def clean_coeff_event(chain, noisy: CoeffEvent, clean_images) -> CoeffEvent:
    """The noise-free target on the noisy event's support: the clean image of every plane
    (``{gid: (wires, ticks)}``, numpy or a tensor) through the chain's transform (no gate, no
    threshold, the dropped finest bands zeroed), read at the noisy coefficients' (band, wire, tau).

    The gather runs on the chain's device: the plane's bands are concatenated along time and
    indexed with (row, column) tensors, so only the selected values come back to the host, not
    the dense bands (~205 MB/event). ``sigma_threshold`` is the NOISY event's: a noise-free
    image's bands are mostly exact zeros, so their own MAD sigma would be 0 and say nothing;
    the sigma that applies to this target is the one its normalisation shares with the input.
    """
    gids = [int(g) for g in noisy.gids]
    for gi, g in enumerate(gids):                     # geometry must match, or the gather misaligns
        if g not in clean_images:
            raise ValueError(f"clean_images missing gid {g} present in the noisy event")
        nw = int(clean_images[g].shape[0])
        if nw != int(noisy.n_wires[gi]):
            raise ValueError(f"gid {g}: clean has {nw} wires, noisy has {int(noisy.n_wires[gi])} "
                             f"— geometry must match")
    vals = np.zeros(len(noisy.value), np.float32)
    col_off = np.concatenate([[0], np.cumsum(np.asarray(noisy.basis.band_lengths, np.int64))])[:-1]
    for g in gids:
        sel = np.nonzero(noisy.plane_gid == g)[0]
        if not len(sel):
            continue
        bands = chain.dwt_rows(clean_images[g])
        rows = noisy.wire[sel].astype(np.int64)
        cols = col_off[noisy.band[sel]] + noisy.tau[sel]
        if hasattr(bands[0], "detach"):               # torch: gather on the device
            import torch
            cat = torch.cat(list(bands), dim=1)
            vals[sel] = cat[torch.as_tensor(rows, device=cat.device),
                            torch.as_tensor(cols, device=cat.device)].float().cpu().numpy()
        else:
            vals[sel] = np.concatenate(list(bands), axis=1)[rows, cols]
    # SHARE the noisy coords, do not copy them: identical by construction (co-supported), never
    # persisted for the clean target (written coords=False), never mutated. Copying them doubled
    # the builder's accumulation (~19.6 MB/event) and got a 1000-event shard OOM-killed at 16 GB.
    return CoeffEvent(band=noisy.band, plane_gid=noisy.plane_gid, wire=noisy.wire, tau=noisy.tau, value=vals,
                      gids=noisy.gids, n_wires=noisy.n_wires, sigma_threshold=noisy.sigma_threshold.copy(),
                      basis=noisy.basis, run=noisy.run, source_file=noisy.source_file, event=noisy.event,
                      wire_valid=noisy.wire_valid)
