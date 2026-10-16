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
    (``{gid: (wires, ticks)}``) through the chain's transform (no gate, no threshold, the dropped
    finest bands zeroed), read at the noisy coefficients' (band, wire, tau)."""
    vals = np.zeros(len(noisy.value), np.float32)
    for g in np.unique(noisy.plane_gid):
        sel = np.nonzero(noisy.plane_gid == g)[0]
        bands = [_host(b) for b in chain.dwt_rows(clean_images[int(g)])]
        for b in np.unique(noisy.band[sel]):
            s2 = sel[noisy.band[sel] == b]
            vals[s2] = bands[int(b)][noisy.wire[s2], noisy.tau[s2]]
    return CoeffEvent(band=noisy.band, plane_gid=noisy.plane_gid, wire=noisy.wire, tau=noisy.tau, value=vals,
                      gids=noisy.gids, n_wires=noisy.n_wires, sigma_threshold=noisy.sigma_threshold, basis=noisy.basis,
                      run=noisy.run, source_file=noisy.source_file, event=noisy.event, wire_valid=noisy.wire_valid)
