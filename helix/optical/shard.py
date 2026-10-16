"""Optical shards: one modality per file, alongside the TPC coefficient shards (helix.core.coeff_io).

Per event, every light system's :meth:`helix.optical.codec.OpticalChain.process` result, encoded
as exact integer streams (lossless for what the codec stored). The identity block matches
coeff_io's, so an event's TPC and optical shards join on (run, source_file, event).

Layout::

    /config              attrs: format, modality = "optical", n_events
    /ident               run (str), source_file (str), event int64, subrun int64, time float64   (n_events,)
    /optical/<system>    attrs: spec_json (the codec spec); wf_offsets (n_events + 1), noise_offsets (n_events + 1)
        per waveform:    channel, t_us, length, n_region, n_cand, n_value
        streams:         region_gap / region_len (uint16): the runs of the marks of the stored blocks;
                         coarse (int32): the window's ceil(length / 2^level) coarse sums, delta-coded;
                         keep_bits: one bit per candidate difference (blocks touching the marks) on
                         every level, packed per waveform; value (int32): the stored differences,
                         levels coarse to fine, ascending within a level
        noise per event and channel: noise_channel, noise_stat, noise_level (n, level)

:func:`read_optical` returns per-waveform coarse sums, differences and stored masks;
:func:`helix.optical.codec.unhaar` decodes; :func:`helix.optical.tokens.shard_tokens` tokenizes.
"""
from __future__ import annotations

import json

import h5py
import numpy as np

FORMAT = "helix-optical-2"
_CHUNK = 1 << 18
_STR = h5py.string_dtype(encoding="utf-8")


def _host(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _segments(lengths, order):
    """Index that reorders concatenated per-row segments (lengths) into ``order``."""
    lengths = np.asarray(lengths, np.int64)
    start = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    ln = lengths[order]
    return np.repeat(start[order] - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln) + np.arange(ln.sum())


def _group_streams(g, level):
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
    Cq = cat(cand)
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
    lev = spec["level"]
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
        c, p = _group_streams(g, lev)
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


class OpticalWriter:
    """Append events (every system's codec result per event) to one optical shard; pending events
    are flushed every ``flush_every``. ``level``: deflate level."""

    def __init__(self, path, level=4, flush_every=8):
        self.f = h5py.File(path, "w")
        self.level, self.flush_every = level, flush_every
        self.f.create_group("config").attrs.update(format=FORMAT, modality="optical", n_events=0)
        self.f.create_group("ident"); self.f.create_group("optical")
        self.pend, self.ident, self.n, self.n_pending = {}, [], 0, 0

    def _append(self, grp, name, arr):
        arr = np.asarray(arr)
        if name not in grp:
            chunks = (min(max(len(arr), 1), _CHUNK),) if arr.ndim == 1 else (64,) + arr.shape[1:]
            kw = {} if arr.dtype.kind in "OSU" else dict(compression="gzip", compression_opts=self.level, shuffle=True)
            grp.create_dataset(name, data=arr, maxshape=(None,) + arr.shape[1:], chunks=chunks, **kw)
            return
        d = grp[name]; n0 = d.shape[0]
        d.resize(n0 + len(arr), axis=0); d[n0:] = arr

    def add(self, results, run="", source_file="", event=0, subrun=0, time=0.0):
        """One event: ``results`` a list of codec results (one per system; a system missing from an
        event is written with no waveforms)."""
        for r in results:
            s = r["system"]; table, data, noise = optical_streams(r)
            if s not in self.pend:
                g = self.f["optical"].create_group(s)
                g.attrs.update(spec_json=json.dumps(r["spec"]))
                self._append(g, "wf_offsets", np.zeros(1, np.int64)); self._append(g, "noise_offsets", np.zeros(1, np.int64))
                self.pend[s] = dict(tables=[], data=[], noise=[], nwf=[], nch=[], total=0, ntotal=0, spec=r["spec"], n=self.n)
                for _ in range(self.n):                            # events before the system first appeared
                    self.pend[s]["nwf"].append(0); self.pend[s]["nch"].append(0)
            p = self.pend[s]
            if r["spec"] != p["spec"]:
                raise ValueError(f"optical {s}: the codec spec changed within a file")
            p["tables"].append(table); p["data"].append(data); p["noise"].append(noise)
            p["nwf"].append(len(table["length"])); p["nch"].append(len(noise["noise_channel"]))
        for s, p in self.pend.items():                              # systems absent from this event
            if s not in {r["system"] for r in results}:
                p["nwf"].append(0); p["nch"].append(0)
        self.ident.append((str(run), str(source_file), int(event), int(subrun), float(time)))
        self.n += 1; self.n_pending += 1
        if self.n_pending >= self.flush_every:
            self.flush()

    def flush(self):
        for s, p in self.pend.items():
            if not p["nwf"]:
                continue
            g = self.f["optical"][s]
            ends = p["total"] + np.cumsum(p["nwf"]); p["total"] = int(ends[-1])
            self._append(g, "wf_offsets", ends.astype(np.int64))
            ne = p["ntotal"] + np.cumsum(p["nch"]); p["ntotal"] = int(ne[-1])
            self._append(g, "noise_offsets", ne.astype(np.int64))
            for part in ("tables", "data", "noise"):
                if p[part]:
                    for k in p[part][0]:
                        self._append(g, k, np.concatenate([t[k] for t in p[part]]))
            p.update(tables=[], data=[], noise=[], nwf=[], nch=[])
        if self.ident:
            run, src, ev, sub, tm = zip(*self.ident)
            gi = self.f["ident"]
            self._append(gi, "run", np.array(run, dtype=_STR))
            self._append(gi, "source_file", np.array(src, dtype=_STR))
            self._append(gi, "event", np.array(ev, np.int64)); self._append(gi, "subrun", np.array(sub, np.int64))
            self._append(gi, "time", np.array(tm, np.float64))
            self.ident = []
        self.f["config"].attrs["n_events"] = self.n
        self.n_pending = 0
        self.f.flush()

    def close(self):
        self.flush(); self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def n_events(path):
    with h5py.File(path, "r") as f:
        return int(f["config"].attrs["n_events"])


def read_optical(path, system, i):
    """Event ``i``'s waveforms of ``system`` -> list of dict(channel, t_us, length, level, roi,
    coarse, details [d_level..d_1], stored [masks], sigma_stat, sigma_level): integer arrays over
    the window padded to a multiple of 2^level; :func:`helix.optical.codec.unhaar` decodes."""
    out = []
    with h5py.File(path, "r") as f:
        g = f["optical"][system]
        spec = json.loads(g.attrs["spec_json"])
        lev = spec["level"]
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
            isq = np.concatenate([np.full(Lw >> j, True) for j in range(lev, 0, -1)])
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
