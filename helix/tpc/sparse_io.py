"""Sparse coefficient files for the configurable TPC chain (:mod:`helix.tpc.chain`) and
the optical codec (:mod:`helix.optical.codec`).

HELIX's coefficient shard (:func:`helix.core.coeff_io.write_coeff_shard`) assumes ONE
basis per event (wavelet, level, band lengths). The SBND chain thresholds planes at
different levels (U level 4, V / Y level 3), so this file records the basis PER PLANE
and stores each plane's coefficients in one of two encodings:

  * ``columns``: HELIX's per-coefficient columns - band uint8, wire int32 (row within
    the plane), tau int32 (index within the band), value float32.
  * ``compact``: the sorted linear index ``wire * sum(band lengths) + band offset + tau``,
    delta-coded from 0 in each event as uint32, and float16 values (~2.5 bytes per
    coefficient on SBND).
All datasets are chunked, byte-shuffled and deflated (``level``), and APPENDED every
``flush_every`` events, so a file of any number of events is written with bounded
memory.

Layout::

    /config   attrs: chain_json, chain_digest, readout, readout_meta, encoding, format
    /planes/<gid>   attrs: tpc, plane, wavelet, level, n_ticks, band_lengths; rows (channel ids)
    /events         run, subrun, event, time (n_events,)
    /coeffs/<gid>   offsets int64 (n_events + 1); sigma float32 (n_events, n_bands);
                    valid uint8 bitmask (n_events, ceil(rows / 8));
                    columns: band, wire, tau, value | compact: dindex, value

gid = tpc * n_planes + plane. :func:`read_event` returns dense band lists per plane.

Optical systems, ``/optical/<system>`` (:mod:`helix.optical.codec`; attrs ``spec_json``,
``format``): per waveform (concatenated over events; ``wf_offsets`` (n_events + 1)):
channel, t_us, length, n_region, n_cand, n_value. Streams, in waveform order:

  * ``region_gap`` / ``region_len`` (uint16): the support runs (gap from the previous run's end);
  * ``coarse`` (int32): the window's ceil(length / 2^level) coarse sums, delta-coded within it;
  * ``keep_bits``: one bit per candidate difference (blocks touching the marks of the stored
    blocks) on every level, packed per waveform;
  * ``value`` (int32): the stored differences, levels coarse to fine, ascending within a level.
Noise per event and channel (``noise_offsets`` (n_events + 1)): noise_channel, noise_stat,
noise_level (n, level). :func:`read_optical` returns per-waveform coarse sums, differences,
stored masks and support.
"""
from __future__ import annotations

import json

import h5py
import numpy as np

from helix.optical import codec

FORMAT = "helix-tpc-sparse-2"
OPTICAL_FORMAT = "optical-haar-1"
_CHUNK = 1 << 18


def _host(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _nonzero(c):
    """(rows, cols, values) of a band's nonzero entries, as host arrays; torch bands are
    reduced on their device, so only the sparse entries cross to the host."""
    if hasattr(c, "detach"):
        idx = c.nonzero(as_tuple=False)
        w, t = idx[:, 0], idx[:, 1]
        return w.cpu().numpy(), t.cpu().numpy(), c[w, t].cpu().numpy()
    w, t = np.nonzero(c)
    return w, t, c[w, t]


def sparse_plane(res):
    """One plane result of :meth:`TPCChain.process_tpc` -> host arrays (band, wire, tau, value)."""
    bs, ws, ts, vs = [], [], [], []
    for bi, c in enumerate(res["bands"]):
        w, t, v = _nonzero(c)
        bs.append(np.full(len(w), bi, np.uint8)); ws.append(w.astype(np.int32)); ts.append(t.astype(np.int32))
        vs.append(v.astype(np.float32))
    return np.concatenate(bs), np.concatenate(ws), np.concatenate(ts), np.concatenate(vs)


def _segments(lengths, order):
    """Index that reorders concatenated per-row segments (lengths) into ``order``."""
    lengths = np.asarray(lengths, np.int64)
    start = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    ln = lengths[order]
    return np.repeat(start[order] - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(ln.sum())


def _group_streams(g, level, q_levels):
    """One group's streams, reduced where the arrays live (torch: on the device; only the
    stored entries cross to the host)."""
    roi, det, st = g["roi"], g["details"], g["stored"]
    m, Lp = roi.shape
    torch_ = hasattr(roi, "detach")
    if torch_:
        import torch
        cat, zeros = (lambda xs: torch.cat(xs, 1)), torch.zeros_like
        ar = lambda n: torch.arange(n, device=roi.device)  # noqa: E731
    else:
        cat, zeros, ar = (lambda xs: np.concatenate(xs, 1)), np.zeros_like, np.arange
    cand = [roi.reshape(m, d.shape[1], -1).any(-1) for d in det]
    Cq = cat([c if level - li <= q_levels else zeros(c) for li, c in enumerate(cand)])
    S = cat(st)
    vals, bits = cat(det)[S], S[Cq]
    ncand, nval = Cq.sum(1), S.sum(1)
    nco = (g["lengths"] + (1 << level) - 1) >> level
    co = g["coarse"]
    dco = co.clone() if torch_ else co.copy()
    dco[:, 1:] -= co[:, :-1]
    ncot = torch.as_tensor(nco, device=roi.device) if torch_ else nco
    cvals = dco[ar(co.shape[1])[None, :] < ncot[:, None]]
    r8 = roi.to(torch.int8) if torch_ else roi.view(np.int8)
    if torch_:
        z = torch.zeros((m, 1), dtype=torch.int8, device=roi.device)
        e = torch.diff(torch.cat([z, r8, z], 1), dim=1)
        rs, s0 = torch.nonzero(e == 1, as_tuple=True); e0 = torch.nonzero(e == -1, as_tuple=True)[1]
        nreg = torch.bincount(rs, minlength=m)
        h = lambda t: t.cpu().numpy()  # noqa: E731
        nreg, rs, s0, e0, ncand, bits, nval, vals, cvals = map(h, (nreg, rs, s0, e0, ncand, bits, nval, vals, cvals))
    else:
        e = np.diff(np.pad(r8, ((0, 0), (1, 1))), axis=1)
        r_, t_ = np.nonzero(e)                                   # starts (+1) and ends (-1) alternate
        up = e[r_, t_] > 0
        rs, s0, e0 = r_[up], t_[up], t_[~up]
        nreg = np.bincount(rs, minlength=m)
    prev_end = np.concatenate([[0], e0[:-1]])
    if len(rs):
        prev_end[np.r_[True, rs[1:] != rs[:-1]]] = 0
    return dict(n_region=nreg, n_cand=ncand, n_value=nval, n_coarse=nco), \
        dict(region_gap=s0 - prev_end, region_len=e0 - s0, keep_bits=bits, value=vals, coarse=cvals)


def optical_streams(res):
    """One system's :meth:`helix.optical.codec.OpticalChain.process` result -> (per-waveform
    table, {stream: array}, per-channel noise), in waveform order."""
    spec = res["spec"]
    lev = ql = spec["level"]
    if not res["groups"]:
        z = lambda dt: np.zeros(0, dt)  # noqa: E731
        table = dict(channel=z(np.int32), t_us=z(np.float64), length=z(np.int32), n_region=z(np.int32),
                     n_cand=z(np.int32), n_value=z(np.int32))
        data = dict(region_gap=z(np.uint16), region_len=z(np.uint16), coarse=z(np.int32), keep_bits=z(np.uint8),
                    value=z(np.int32))
        return table, data, dict(noise_channel=z(np.int32), noise_stat=z(np.float32),
                                 noise_level=np.zeros((0, lev), np.float32))
    rows, cnts, parts = [], [], []
    for g in res["groups"]:
        c, p = _group_streams(g, lev, ql)
        rows.append(np.asarray(g["index"])); cnts.append(c); parts.append(p)
    rows = np.concatenate(rows); order = np.argsort(rows, kind="stable")
    cnt = {k: np.concatenate([c[k] for c in cnts]).astype(np.int64) for k in cnts[0]}
    seg = dict(region_gap="n_region", region_len="n_region", keep_bits="n_cand", value="n_value", coarse="n_coarse")
    data = {k: np.concatenate([p[k] for p in parts])[_segments(cnt[n], order)] for k, n in seg.items()}
    nc = cnt["n_cand"][order]
    kb = data.pop("keep_bits").astype(bool)
    pos = np.arange(len(kb)) - np.repeat(np.concatenate([[0], np.cumsum(nc)[:-1]]), nc)    # bit index within waveform
    byte0 = np.concatenate([[0], np.cumsum((nc + 7) // 8)[:-1]])
    data["keep_bits"] = np.bincount(np.repeat(byte0, nc)[kb] + pos[kb] // 8, weights=(128 >> (pos[kb] % 8)),
                                    minlength=int(((nc + 7) // 8).sum())).astype(np.uint8)
    data.update(region_gap=data["region_gap"].astype(np.uint16), region_len=data["region_len"].astype(np.uint16),
                value=data["value"].astype(np.int32), coarse=data["coarse"].astype(np.int32))
    table = dict(channel=np.asarray(res["channel"], np.int32), t_us=np.asarray(res["t_us"], np.float64),
                 length=np.asarray(res["length"], np.int32),
                 **{k: cnt[k][order].astype(np.int32) for k in ("n_region", "n_cand", "n_value")})
    nz = res["noise"]
    noise = dict(noise_channel=np.asarray(nz["channel"], np.int32), noise_stat=np.asarray(nz["stat"], np.float32),
                 noise_level=np.asarray(nz["level"], np.float32))
    return table, data, noise


class Writer:
    """Append events to one file (bounded memory: pending events are flushed every
    ``flush_every``). ``level``: deflate level (default 4)."""

    def __init__(self, path, chain, encoding="compact", level=4, flush_every=8):
        if encoding not in ("columns", "compact"):
            raise ValueError(encoding)
        self.f = h5py.File(path, "w")
        self.chain, self.encoding, self.level, self.flush_every = chain, encoding, level, flush_every
        self.n_planes = len(chain.readout.planes)
        c = self.f.create_group("config")
        from helix.tpc.chain import config_json
        c.attrs.update(chain_json=config_json(chain.cfg), chain_digest=chain.digest, readout=chain.readout.path,
                       readout_meta=json.dumps(chain.readout.meta), encoding=encoding, format=FORMAT,
                       deflate_level=level)
        for g in ("planes", "coeffs", "events"):
            self.f.create_group(g)
        self.pend, self.opend, self.meta = {}, {}, []
        self.n, self.n_pending = 0, 0

    # ------------------------------------------------------------------ storage
    def _append(self, grp, name, arr):
        arr = np.asarray(arr)
        if name not in grp:
            chunks = (min(max(len(arr), 1), _CHUNK),) if arr.ndim == 1 else (64,) + arr.shape[1:]
            grp.create_dataset(name, data=arr, maxshape=(None,) + arr.shape[1:], chunks=chunks,
                               compression="gzip", compression_opts=self.level, shuffle=True)
            return
        d = grp[name]
        n0 = d.shape[0]
        d.resize(n0 + len(arr), axis=0)
        d[n0:] = arr

    # ------------------------------------------------------------------ TPC
    def _init_plane(self, gid, r):
        """``r``: a plane result or the ``info`` dict of :meth:`prepare`."""
        g = self.f["planes"].create_group(str(gid))
        g.attrs.update(tpc=int(r["tpc"]), plane=r["plane"], wavelet=r["wavelet"], level=int(r["level"]),
                       n_ticks=int(r["n_ticks"]), band_lengths=np.asarray(r["band_lengths"], np.int64))
        g.create_dataset("rows", data=np.asarray(r["rows"], np.int32))
        cg = self.f["coeffs"].create_group(str(gid))
        self._append(cg, "offsets", np.zeros(1, np.int64))
        self.pend[gid] = dict(parts=[], counts=[], sigma=[], valid=[], total=0, L=int(sum(r["band_lengths"])),
                              off=np.concatenate([[0], np.cumsum(r["band_lengths"])[:-1]]).astype(np.int64))

    def prepare(self, plane_results, optical_results=(), run=0, subrun=0, event=0, time=0.0):
        """Everything :meth:`add_prepared` needs, as host arrays (device -> host sparse
        extraction happens here). Split from the write so a writer thread can compress
        and write while the next event runs on the device."""
        planes = []
        for r in plane_results:
            gid = int(r["tpc"]) * self.n_planes + self.chain.readout.planes.index(r["plane"])
            info = dict(tpc=int(r["tpc"]), plane=r["plane"], wavelet=r["wavelet"], level=int(r["level"]),
                        n_ticks=int(r["n_ticks"]), band_lengths=list(r["band_lengths"]), rows=np.asarray(r["rows"]))
            planes.append((gid, info, sparse_plane(r), np.asarray(r["sigma_per_band"], np.float32),
                           np.packbits(_host(r["valid"]).astype(bool))))
        optical = [(o["system"], o["spec"], optical_streams(o)) for o in optical_results]
        return dict(planes=planes, optical=optical, meta=(run, subrun, event, time))

    def add_prepared(self, prep):
        for gid, info, (band, wire, tau, val), sigma, valid in prep["planes"]:
            if gid not in self.pend:
                self._init_plane(gid, info)
            b = self.pend[gid]
            if self.encoding == "compact":
                lin = wire.astype(np.int64) * b["L"] + b["off"][band] + tau
                o = np.argsort(lin, kind="stable")
                d = np.diff(np.concatenate([[0], lin[o]])).astype(np.uint32)
                b["parts"].append((d, val[o].astype(np.float16)))
            else:
                b["parts"].append((band, wire, tau, val))
            b["counts"].append(len(val))
            b["sigma"].append(sigma)
            b["valid"].append(valid)
        for system, spec, streams in prep["optical"]:
            self._add_optical_streams(system, spec, streams)
        self.meta.append(prep["meta"])
        self.n += 1; self.n_pending += 1
        if self.n_pending >= self.flush_every:
            self.flush()

    def add(self, plane_results, run=0, subrun=0, event=0, time=0.0):
        """Add one event's TPC plane results (optical: :meth:`add_optical` for the same event)."""
        self.add_prepared(self.prepare(plane_results, (), run, subrun, event, time))

    # ------------------------------------------------------------------ optical
    def add_optical(self, res):
        """One system's :meth:`helix.optical.codec.OpticalChain.process` result for the current
        event (call once per system per event, with :meth:`add` of the same event)."""
        self._add_optical_streams(res["system"], res["spec"], optical_streams(res))

    def _add_optical_streams(self, s, spec, streams):
        table, data, noise = streams
        if s not in self.opend:
            g = self.f.require_group("optical").create_group(s)
            g.attrs.update(spec_json=json.dumps(spec), format=OPTICAL_FORMAT)
            self._append(g, "wf_offsets", np.zeros(1, np.int64))
            self._append(g, "noise_offsets", np.zeros(1, np.int64))
            self.opend[s] = dict(tables=[], data=[], noise=[], nwf=[], nch=[], total=0, ntotal=0, spec=spec)
        p = self.opend[s]
        if spec != p["spec"]:
            raise ValueError(f"optical {s}: the codec spec changed within a file")
        p["tables"].append(table); p["data"].append(data); p["noise"].append(noise)
        p["nwf"].append(len(table["length"])); p["nch"].append(len(noise["noise_channel"]))

    # ------------------------------------------------------------------ flush / close
    def flush(self):
        for gid, b in self.pend.items():
            if not b["counts"]:
                continue
            g = self.f["coeffs"][str(gid)]
            ends = b["total"] + np.cumsum(b["counts"])
            b["total"] = int(ends[-1])
            self._append(g, "offsets", ends.astype(np.int64))
            self._append(g, "sigma", np.stack(b["sigma"]))
            self._append(g, "valid", np.stack(b["valid"]))
            names = ("dindex", "value") if self.encoding == "compact" else ("band", "wire", "tau", "value")
            for j, name in enumerate(names):
                self._append(g, name, np.concatenate([p[j] for p in b["parts"]]))
            b.update(parts=[], counts=[], sigma=[], valid=[])
        for s, p in self.opend.items():
            if not p["nwf"]:
                continue
            g = self.f["optical"][s]
            ends = p["total"] + np.cumsum(p["nwf"])
            p["total"] = int(ends[-1])
            self._append(g, "wf_offsets", ends.astype(np.int64))
            ne = p["ntotal"] + np.cumsum(p["nch"])
            p["ntotal"] = int(ne[-1])
            self._append(g, "noise_offsets", ne.astype(np.int64))
            for part in ("tables", "data", "noise"):
                for k in p[part][0]:
                    self._append(g, k, np.concatenate([t[k] for t in p[part]]))
            p.update(tables=[], data=[], noise=[], nwf=[], nch=[])
        if self.meta:
            m = np.array(self.meta, dtype=[("run", "i8"), ("subrun", "i8"), ("event", "i8"), ("time", "f8")])
            for k in m.dtype.names:
                self._append(self.f["events"], k, m[k])
            self.meta = []
        self.n_pending = 0
        self.f.flush()

    def close(self):
        self.flush()
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_event(path, i):
    """Event ``i`` -> {gid: dict(tpc, plane, wavelet, level, n_ticks, rows, valid, sigma,
    bands)} with dense float32 bands (rows: channel ids; valid: per row)."""
    out = {}
    with h5py.File(path, "r") as f:
        enc = f["config"].attrs["encoding"]
        for gid, g in f["coeffs"].items():
            pa = f["planes"][gid].attrs
            bl = np.asarray(pa["band_lengths"], np.int64)
            rows = f["planes"][gid]["rows"][:]
            a, b = g["offsets"][i], g["offsets"][i + 1]
            bands = [np.zeros((len(rows), n), np.float32) for n in bl]
            if enc == "compact":
                lin = np.cumsum(g["dindex"][a:b].astype(np.int64))       # deltas restart at 0 in each event
                val = g["value"][a:b].astype(np.float32)
                L = int(bl.sum()); off = np.concatenate([[0], np.cumsum(bl)[:-1]])
                wire, rem = lin // L, lin % L
                band = np.searchsorted(off, rem, side="right") - 1
                tau = rem - off[band]
            else:
                band, wire, tau = g["band"][a:b], g["wire"][a:b], g["tau"][a:b]
                val = g["value"][a:b]
            for bi in range(len(bl)):
                m = band == bi
                bands[bi][wire[m], tau[m]] = val[m]
            valid = np.unpackbits(g["valid"][i])[:len(rows)].astype(bool)
            out[int(gid)] = dict(tpc=int(pa["tpc"]), plane=str(pa["plane"]), wavelet=str(pa["wavelet"]),
                                 level=int(pa["level"]), n_ticks=int(pa["n_ticks"]), rows=rows, valid=valid,
                                 sigma=g["sigma"][i].astype(np.float32), bands=bands)
    return out


def coeff_event(path, i, sigma_norm=2.6):
    """Event ``i`` -> :class:`helix.core.coeff_event.CoeffEvent`, HELIX's corpus atom.

    ``wire`` is the PHYSICAL wire number (the readout map's), not the row: a plane's
    uninstrumented wires (SBND: Y wires 832-837 of both TPCs) are gaps, so cells keep
    their geometry. ``n_wires`` = the plane's physical wire count; ``wire_valid`` is
    False for every wire without a valid row in this event (not instrumented, dead,
    noisy, baseline step): the tokenizer gives those slots no target."""
    from helix.core.coeff_event import CoeffEvent
    from helix.core.provenance import BasisDescriptor
    from helix.tpc.readout import load_readout
    with h5py.File(path, "r") as f:
        cfg = json.loads(f["config"].attrs["chain_json"])
        ev = f["events"]
        run, event = int(ev["run"][i]), int(ev["event"][i])
    ro = load_readout(cfg["readout"])
    planes = read_event(path, i)
    gids = np.array(sorted(planes), np.int32)
    n_wires = np.array([int(ro.wire[(ro.tpc == planes[g]["tpc"]) & (ro.plane == ro.planes.index(planes[g]["plane"]))]
                            .max()) + 1 for g in gids], np.int32)
    wire_valid = np.zeros((len(gids), int(n_wires.max())), bool)
    cols = {k: [] for k in ("band", "plane_gid", "wire", "tau", "value")}
    for gi, g in enumerate(gids):
        p = planes[g]
        w_phys = ro.wire[p["rows"]]
        wire_valid[gi, w_phys[p["valid"]]] = True
        for b, c in enumerate(p["bands"]):
            r, t = np.nonzero(c)
            cols["band"].append(np.full(len(r), b, np.uint8)); cols["plane_gid"].append(np.full(len(r), g, np.int32))
            cols["wire"].append(w_phys[r].astype(np.int32)); cols["tau"].append(t.astype(np.int32))
            cols["value"].append(c[r, t].astype(np.float32))
    p0 = planes[int(gids[0])]
    bl = tuple(int(c.shape[1]) for c in p0["bands"])
    basis = BasisDescriptor(wavelet=p0["wavelet"], level=p0["level"], mode=cfg["dwt"]["mode"],
                            n_ticks_raw=p0["n_ticks"], pad=bl[0] * 2 ** p0["level"] - p0["n_ticks"], band_lengths=bl,
                            removal=dict(cfg["coherent"], readout=cfg["readout"]), threshold=dict(cfg["threshold"]),
                            sigma_norm=sigma_norm)
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    return CoeffEvent(**cat, gids=gids, n_wires=n_wires,
                      sigma_threshold=np.stack([planes[int(g)]["sigma"] for g in gids]).astype(np.float32),
                      basis=basis, run=str(run), source_file=str(path), event=event, wire_valid=wire_valid)


def read_optical(path, system, i):
    """Event ``i``'s waveforms of ``system`` -> list of dict(channel, t_us, length, level, roi,
    coarse, details [d_level..d_1], stored [masks], sigma_stat, sigma_level): integer arrays over
    the window padded to a multiple of 2^level; :func:`helix.optical.codec.unhaar` decodes."""
    out = []
    with h5py.File(path, "r") as f:
        g = f["optical"][system]
        spec = json.loads(g.attrs["spec_json"])
        lev = ql = spec["level"]
        a, b = int(g["wf_offsets"][i]), int(g["wf_offsets"][i + 1])
        cnt = {k: g[k][:b].astype(np.int64) for k in ("n_region", "n_cand", "n_value", "length")}
        nco = (cnt["length"] + (1 << lev) - 1) >> lev
        o = {k: int(v[:a].sum()) for k, v in dict(r=cnt["n_region"], v=cnt["n_value"], c=nco,
                                                  b=(cnt["n_cand"] + 7) // 8).items()}
        e = {k: o[k] + int(v[a:b].sum()) for k, v in dict(r=cnt["n_region"], v=cnt["n_value"], c=nco,
                                                         b=(cnt["n_cand"] + 7) // 8).items()}
        gap, rl = g["region_gap"][o["r"]:e["r"]].astype(np.int64), g["region_len"][o["r"]:e["r"]].astype(np.int64)
        val, co = g["value"][o["v"]:e["v"]].astype(np.int64), g["coarse"][o["c"]:e["c"]].astype(np.int64)
        bits = g["keep_bits"][o["b"]:e["b"]]
        n0, n1 = int(g["noise_offsets"][i]), int(g["noise_offsets"][i + 1])
        nch = g["noise_channel"][n0:n1]; nst = g["noise_stat"][n0:n1]; nlv = g["noise_level"][n0:n1]
        noise = {int(c): (float(s_), lv) for c, s_, lv in zip(nch, nst, nlv)}
        rp = vp = cp = bp = 0
        for w in range(a, b):
            n = int(cnt["length"][w]); Lw = int(nco[w]) << lev
            m = int(cnt["n_region"][w]); roi = np.zeros(Lw, bool)
            st = np.cumsum(np.r_[0, rl[rp:rp + m - 1]] + gap[rp:rp + m]) if m else np.zeros(0, np.int64)
            for s0, ln in zip(st, rl[rp:rp + m]):
                roi[s0:s0 + ln] = True
            rp += m
            cand = [roi.reshape(Lw >> j, 1 << j).any(-1) for j in range(lev, 0, -1)]
            isq = np.concatenate([np.full(Lw >> j, j <= ql) for j in range(lev, 0, -1)])
            C = np.concatenate(cand)
            nc = int(cnt["n_cand"][w]); kb = np.unpackbits(bits[bp:bp + (nc + 7) // 8])[:nc].astype(bool)
            bp += (nc + 7) // 8
            S = C.copy(); qpos = np.nonzero(C & isq)[0]; S[qpos[~kb]] = False
            k = int(cnt["n_value"][w]); flat = np.zeros(len(C), np.int64); flat[S] = val[vp:vp + k]; vp += k
            cut = np.cumsum([0] + [Lw >> j for j in range(lev, 0, -1)])
            c = np.zeros(Lw >> lev, np.int64); c[:int(nco[w])] = np.cumsum(co[cp:cp + int(nco[w])]); cp += int(nco[w])
            ch = int(g["channel"][w]); sstat, slev = noise.get(ch, (np.nan, np.full(lev, np.nan, np.float32)))
            out.append(dict(channel=ch, t_us=float(g["t_us"][w]), length=n, level=lev, roi=roi, coarse=c,
                            details=[flat[cut[q]:cut[q + 1]] for q in range(lev)],
                            stored=[S[cut[q]:cut[q + 1]] for q in range(lev)], sigma_stat=sstat,
                            sigma_level=np.asarray(slev, np.float32)))
    return out
