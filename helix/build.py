"""Build shards: the events of one source through the TPC chain and the optical codec.

    python -m helix.build {sim,sbnd,goop} --input FILE [FILE ...] --output DIR/PREFIX --device cuda \\
        [--first 0] [--count N | --spread N] [--events-per-shard 100] [--shard-start K] \\
        [--norm-sigma TABLE.npy | --calibrate TABLE.npy] [--config chain.json|yaml] \\
        [--optical-config ...] [--no-optical] [--workers 4] [--deflate 4] [--events-log F] [--profile]

A source only reads: per event its identity, the TPC input and the optical waveforms.

* ``sim``: JAXTPC sensor files (one run). The planes are densified, noised (the colored
  incoherent + coherent model, :mod:`helix.tpc.dense_ops`, seeded per event) and digitized on
  the device; the noise-free planes are the clean target. Chain: :func:`helix.tpc.chain.sim_config`.
* ``sbnd``: extracted SBND files (:mod:`helix.tpc.io_sbnd`, decoded in worker processes): the
  readout (adc, pedestal, present per channel), PMT and X-ARAPUCA waveforms. Chain: SBND_DEFAULT.
* ``goop``: goop light files, optical only (the codec's GOOP specs and responses); goop's photon
  counts per PMT go into the optical shard as ``truth/pe_counts``.

Everything after the read is shared. Per ``--events-per-shard`` events, shard k
(``--shard-start`` + 0, 1, ...): ``PREFIX_coeff_k.h5`` (:mod:`helix.core.coeff_io`; the dataset
name is PREFIX's last component), ``PREFIX_coeff_clean_k.h5`` (co-supported clean target, sim),
``PREFIX_optical_k.h5`` (:mod:`helix.optical.shard`), joined on (run, source_file, event).
``norm_sigma`` is ``--norm-sigma`` (one table for a whole corpus) or else each shard's own;
``--calibrate TABLE`` writes the table of the selected events and no shards. A JSON summary
(timing, storage, compression) is printed last.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

SOURCES = ("sim", "sbnd", "goop")


def _load_config(path):
    if not path:
        return None
    txt = open(path).read()
    if path.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)


def normalization_table(events) -> np.ndarray:
    """``norm_sigma`` (n_gid, n_bands) = the mean over ``events`` (CoeffEvents) of their
    per-event threshold sigma."""
    tab = np.stack([e.sigma_threshold for e in events]).mean(axis=0).astype(np.float32)
    if tab.size and float(np.nanmax(tab)) <= 0.0:
        raise ValueError("norm_sigma is all zero: no event carried a threshold sigma")
    return tab


def _sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ------------------------------------------------------------------------------------- sources
class SimSource:
    """JAXTPC sensor files of one run: noisy planes (on the device) and the clean target."""

    systems = ()

    def __init__(self, files, device, geom="cubic_wireplane_geometry.json", npz=None):
        from helix.tpc.geometry import _resolve, load_plane_registry
        from helix.tpc.io import list_events, sensor_layout
        from helix.paths import packaged, resolve
        self.files, self.device = list(files), device
        self.layouts = {f: sensor_layout(f) for f in self.files}
        nw = {json.dumps(np.asarray(lay.num_wires).tolist()) for lay in self.layouts.values()}
        if any(lay.num_wires is None for lay in self.layouts.values()) or len(nw) != 1:
            raise SystemExit("sim: every input file needs the same /config/num_wires layout")
        self.num_wires = self.layouts[self.files[0]].num_wires
        self.run = Path(self.files[0]).parent.name
        self.index = [(f, e) for f in self.files for e in list_events(f)]
        if npz is None:                      # JAXTPC's spectrum when a checkout is declared, else helix's copy
            jx, _ = resolve("HELIX_JAXTPC_ROOT")
            cand = Path(jx) / "config" / "noise_spectrum.npz" if jx else None
            npz = str(cand) if cand and cand.exists() else str(packaged("noise_spectrum.npz"))
        z = np.load(npz, allow_pickle=True)
        self.spectrum = (z["spectrum_freqs_hz"], z["spectrum_shape"])
        self.reg = load_plane_registry(geom)
        try:
            geom_path = str(_resolve(geom))
        except Exception:
            geom_path = geom
        self.npz, self.geom_path = npz, geom_path

    def __len__(self):
        return len(self.index)

    def chain_config(self, overrides):
        from helix.tpc.chain import sim_config
        return sim_config(self.num_wires, overrides)

    def bind(self, chain):
        self.group_size = int(chain.cfg["readout"]["group_size"])

    def seed(self, src, ev):
        """The event's noise seed: run, file and event, so no two events share a realisation."""
        key = f"{self.run}/{Path(src).name}/ev{ev}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") & 0xFFFFFFFF

    def events(self, indices, workers):
        import torch
        from pimm_data.dense_ops import densify
        from helix.tpc import dense_ops
        from helix.tpc.io import read_sensor_event_coo
        dev = self.device or "cpu"
        for k in indices:
            path, ev = self.index[k]
            layout = self.layouts[path]
            seed = self.seed(path, ev)
            w_l, t_l, v_l, p_l, gmap = [], [], [], [], {}
            for label, (w, t, v, _, _) in read_sensor_event_coo(path, ev, layout).items():
                gid = layout.plane_gid(label)
                gmap[gid] = label
                w_l.append(torch.as_tensor(np.asarray(w), dtype=torch.long, device=dev))
                t_l.append(torch.as_tensor(np.asarray(t), dtype=torch.long, device=dev))
                v_l.append(torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev))
                p_l.append(torch.full((len(w),), gid, dtype=torch.long, device=dev))
            wire = torch.cat(w_l)
            grids = densify(wire, torch.cat(t_l), torch.cat(v_l), torch.cat(p_l),
                            torch.tensor([wire.numel()], dtype=torch.long, device=dev), self.reg)
            clean = {int(g): x[0].clone() for g, x in grids.items()}
            grids = dense_ops.add_intrinsic_noise(grids, self.reg, seeds=[seed], incoherent=True, coherent=True,
                                                  series_spectrum=self.spectrum, group_size=self.group_size)
            grids = dense_ops.digitize(grids, {int(g): layout.pedestal(gmap[int(g)]) for g in grids}, n_bits=12)
            yield dict(ident=dict(run=self.run, source_file=Path(path).name, event=int(ev), noise_seed=seed),
                       planes={int(g): x[0] for g, x in grids.items()}, clean=clean)

    def shard_attrs(self):
        """What the coeff shards record about the input beyond the chain: the noise and its inputs."""
        noise = dict(kind="colored", incoherent=True, coherent=True, group_size=self.group_size,
                     spectrum=os.path.basename(self.npz), mode="serial", backend="torch")
        prov = dict(geom=os.path.basename(self.geom_path), geom_sha256=_sha256(self.geom_path),
                    spectrum=os.path.basename(self.npz), spectrum_sha256=_sha256(self.npz),
                    seed_formula="blake2b(run/source_file/ev{n})")
        return dict(noise=noise, provenance=prov)


class SBNDSource:
    """Extracted SBND files: the TPC readout and the PMT / X-ARAPUCA waveforms."""

    systems = ("pmt", "xarapuca")

    def __init__(self, files, device, optical=True):
        from helix.tpc.io_sbnd import SBNDEvents
        self.files, self.device = sorted(files), device
        self.systems = SBNDSource.systems if optical else ()
        self.n = len(SBNDEvents(self.files))

    def __len__(self):
        return self.n

    def chain_config(self, overrides):
        from helix.tpc.chain import make_config
        return make_config(overrides)

    def bind(self, chain):
        pass

    def optical_chain(self, overrides):
        from helix.optical.codec import OpticalChain
        return OpticalChain(overrides, device=self.device)

    def events(self, indices, workers):
        from helix.tpc.io_sbnd import loader
        for ev in loader(self.files, workers=workers, indices=indices, optical_kinds=self.systems):
            m = ev["meta"]
            yield dict(ident=dict(run=str(m["run"]), source_file=Path(m["file"]).name, event=int(m["event"]),
                                  subrun=int(m["subrun"]), time=float(m["time"])),
                       readout=(ev["adc"], ev["pedestal"], ev["present"]),
                       optical={s: ev[s] for s in self.systems})

    def shard_attrs(self):
        return dict(noise=None, provenance=None)


class GoopSource:
    """goop light files: both sides' stored chunks per event (channel = pmt_id, +1000 west)."""

    systems = ("goop",)

    def __init__(self, files, device):
        import h5py
        self.device = device
        self.index = []
        for p in files:
            with h5py.File(p, "r") as f:
                self.index += [(p, k) for k in sorted(k for k in f.keys() if k.startswith("event_"))]

    def __len__(self):
        return len(self.index)

    def optical_chain(self, overrides):
        from helix.optical.codec import GOOP, OpticalChain
        return OpticalChain(overrides or GOOP, device=self.device, templates="goop")

    @staticmethod
    def waveforms(f, key):
        """One event's stored chunks: dict(channel, t_us, length, offset, buf) and pe_counts
        (2, n_pmt) east / west."""
        chans, t0, lens, bufs, pe = [], [], [], [], []
        for si, side in enumerate(("east", "west")):
            g = f[key][side]
            o = g["offsets"][:]
            chans.append(g["pmt_id"][:].astype(np.int32) + 1000 * si)
            t0.append(g["t0_ns"][:] / 1e3)
            lens.append(np.diff(o)); bufs.append(g["adc"][:])
            pe.append(f[key][f"pe_counts_{side}"][:] if f"pe_counts_{side}" in f[key] else np.zeros(0))
        L = np.concatenate(lens).astype(np.int64)
        wf = dict(channel=np.concatenate(chans), t_us=np.concatenate(t0), length=L,
                  offset=np.r_[0, np.cumsum(L)[:-1]].astype(np.int64), buf=np.concatenate(bufs))
        n = max(len(p) for p in pe)
        return wf, np.stack([np.pad(p, (0, n - len(p))) for p in pe]) if n else np.zeros((2, 0))

    def events(self, indices, workers):
        import h5py
        files = {}
        for k in indices:
            path, key = self.index[k]
            f = files.get(path) or files.setdefault(path, h5py.File(path, "r"))
            wf, pe = self.waveforms(f, key)
            yield dict(ident=dict(run=Path(path).stem, source_file=Path(path).name, event=int(key.split("_")[1])),
                       optical={"goop": wf}, truth=dict(pe_counts=pe))
        for f in files.values():
            f.close()

    def shard_attrs(self):
        return dict(noise=None, provenance=None)


def make_source(name, files, device, optical=True):
    if name == "sim":
        return SimSource(files, device)
    if name == "sbnd":
        return SBNDSource(files, device, optical=optical)
    if name == "goop":
        return GoopSource(files, device)
    raise ValueError(f"unknown source {name!r}; one of {SOURCES}")


# ------------------------------------------------------------------------------------- shards
def _stack_padded(arrays):
    """Stack arrays of one rank, each axis zero-padded to the largest."""
    shape = np.max([a.shape for a in arrays], axis=0)
    return np.stack([np.pad(a, [(0, s - n) for s, n in zip(shape, a.shape)]) for a in arrays])


class ShardWriter:
    """Shard k of PREFIX per ``events_per_shard`` events: coeff (+ clean) and optical files."""

    def __init__(self, prefix, *, events_per_shard, shard_start, norm, attrs, deflate):
        self.prefix, self.eps, self.k, self.norm, self.attrs, self.deflate = prefix, events_per_shard, shard_start, \
            norm, attrs, deflate
        self.dataset = Path(prefix).name
        self.outputs = []
        self._reset()

    def _reset(self):
        self.noisy, self.clean, self.seeds, self.truth, self.ow, self.n = [], [], [], {}, None, 0

    def add(self, ce, clean, ores, ident, truth):
        from helix.optical.shard import OpticalWriter
        if ce is not None:
            self.noisy.append(ce); self.seeds.append(ident.get("noise_seed", -1))
            if clean is not None:
                self.clean.append(clean)
        if ores:
            if self.ow is None:
                self.ow = OpticalWriter(self.path("optical") + ".part", level=self.deflate)
            self.ow.add(ores, **{k: v for k, v in ident.items() if k != "noise_seed"})
        for name, arr in (truth or {}).items():
            self.truth.setdefault(name, []).append(np.asarray(arr))
        self.n += 1
        if self.n >= self.eps:
            self.close()

    def path(self, kind, k=None):
        return f"{self.prefix}_{kind}_{self.k if k is None else k:04d}.h5"

    def _done(self, kind):
        """Shards are written as ``.part`` and renamed whole, so a file with the final name is complete."""
        os.replace(self.path(kind) + ".part", self.path(kind))
        self.outputs.append(self.path(kind))

    def close(self):
        from helix.core.coeff_io import audit_shard, write_coeff_shard
        if self.noisy:
            if self.norm is not None:
                want = (len(self.noisy[0].gids), self.noisy[0].sigma_threshold.shape[1])
                if self.norm.shape != want or float(np.nanmax(self.norm)) <= 0.0:
                    raise ValueError(f"norm_sigma: shape {self.norm.shape} (expected {want}), all zero or not")
            kw = dict(dataset_name=self.dataset, file_index=self.k,
                      norm_sigma=normalization_table(self.noisy) if self.norm is None else self.norm,
                      noise=self.attrs["noise"], provenance=self.attrs["provenance"],
                      noise_seeds=self.seeds if any(s >= 0 for s in self.seeds) else None)
            if self.clean:
                p = self.path("coeff_clean") + ".part"
                write_coeff_shard(p, self.clean, coords=False, **kw); audit_shard(p); self._done("coeff_clean")
            p = self.path("coeff") + ".part"
            write_coeff_shard(p, self.noisy, **kw); audit_shard(p); self._done("coeff")
        if self.ow is not None:
            for name, arrs in self.truth.items():
                self.ow.f.create_dataset(f"truth/{name}", data=_stack_padded(arrs))
            self.ow.close(); self._done("optical")
        if self.n:
            self.k += 1
        self._reset()


# ------------------------------------------------------------------------------------- build
def select(n, first=0, count=None, spread=None):
    """Event indices: ``count`` from ``first`` (all by default), or ``spread`` evenly over all."""
    if spread:
        return sorted(set(np.linspace(0, n - 1, spread).astype(int).tolist()))
    return list(range(min(first, n), n if count is None else min(n, first + count)))


def build(source, prefix, *, indices, chain=None, ochain=None, events_per_shard=100, shard_start=0,
          norm_sigma=None, calibrate=None, workers=4, deflate=4, events_log=None):
    """Run ``indices`` of ``source`` through ``chain`` (TPC; None: optical only) and ``ochain``
    (optical; None: TPC only) into shards at ``prefix`` (none with ``calibrate``: the path the
    normalization table of the events is saved to). Returns the summary dict."""
    from helix.tpc import chain_coeff
    systems = tuple(source.systems) if ochain is not None else ()
    if chain is not None:
        source.bind(chain)
    device = (chain or ochain).device
    cuda = str(device).startswith("cuda")
    import torch
    sync = torch.cuda.current_stream().synchronize if cuda else (lambda: None)
    # the optical codec is host-heavy (many small kernels, Python per group): on CUDA, with a TPC
    # chain, it runs in a thread on its own stream, overlapping the TPC chain of the same event
    pool = ostream = None
    if cuda and systems and chain is not None:
        import concurrent.futures
        pool = concurrent.futures.ThreadPoolExecutor(1); ostream = torch.cuda.Stream()

    def optical(ev):
        if not systems:
            return []
        if ostream is None:
            return [ochain.process(s, ev["optical"][s]) for s in systems]
        with torch.cuda.stream(ostream):
            out = [ochain.process(s, ev["optical"][s]) for s in systems]
        ostream.synchronize()
        return out

    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    attrs = source.shard_attrs()
    if attrs["provenance"] is not None or chain is not None:      # the GPU generation is part of a build
        attrs["provenance"] = dict(attrs["provenance"] or {},
                                   device=f"torch:{torch.cuda.get_device_name()}" if cuda else "torch:cpu")
    w = ShardWriter(prefix, events_per_shard=events_per_shard, shard_start=shard_start, norm=norm_sigma,
                    attrs=attrs, deflate=deflate)
    # shards are written in a writer thread, overlapping the next event's device work
    q, wstat = queue.Queue(maxsize=4), dict(busy=0.0, error=None)

    def writer():
        while (item := q.get()) is not None:
            tw = time.perf_counter()
            try:
                w.add(*item)
            except Exception as e:                                  # surfaced after the loop
                wstat["error"] = e
            wstat["busy"] += time.perf_counter() - tw

    th = threading.Thread(target=writer, daemon=True)
    if calibrate is None:
        th.start()
    T = dict(wait=0.0, chain=0.0, optical=0.0, extract=0.0)
    okept, osamp = {s: 0 for s in systems}, {s: 0 for s in systems}
    kept, cal, n_done = [], [], 0
    elog = open(events_log, "w") if events_log else None
    t_start = t = time.perf_counter()
    for ev in source.events(indices, workers):
        t1 = time.perf_counter(); T["wait"] += t1 - t
        fut = pool.submit(optical, ev) if pool is not None else None
        res = None
        if chain is not None:
            res = chain.process_event(*ev["readout"]) if "readout" in ev else chain.process_planes(ev["planes"])
        sync(); t2 = time.perf_counter(); T["chain"] += t2 - t1
        ores = fut.result() if fut is not None else optical(ev)
        sync(); t3 = time.perf_counter(); T["optical"] += t3 - t2
        idn = ev["ident"]
        ce = clean = None
        if res is not None:
            ce = chain_coeff.coeff_event(chain, res, run=idn["run"], source_file=idn["source_file"], event=idn["event"])
            if ev.get("clean") is not None and calibrate is None:
                clean = chain_coeff.clean_coeff_event(chain, ce, ev["clean"])
            kept.append(sum(int(r["n_kept"]) for r in res))
        if calibrate is None:
            q.put((ce, clean, ores, idn, ev.get("truth")))
        else:
            cal.append(ce)
        for r in ores:
            okept[r["system"]] += r["n_kept"]; osamp[r["system"]] += r["n_samples"]
        t = time.perf_counter(); T["extract"] += t - t3
        if elog is not None:
            elog.write(json.dumps(dict(**{k: idn[k] for k in ("run", "event")}, chain_s=t2 - t1, optical_s=t3 - t2,
                                       tpc={f"{r['tpc']}{r['plane']}": int(r["n_kept"]) for r in res or []},
                                       optical={r["system"]: dict(kept=int(r["n_kept"]), samples=int(r["n_samples"]),
                                                                  waveforms=int(len(r["length"]))) for r in ores})) + "\n")
        n_done += 1
        if n_done <= 3 or n_done % 50 == 0:
            nk = kept[-1] if res is not None else sum(int(r["n_kept"]) for r in ores)
            print(f"event {idn['run']}:{idn['event']} kept {nk} chain {t2 - t1:.3f}s", flush=True)
    tc = time.perf_counter()
    if calibrate is None:
        q.put(None); th.join(); w.close()
    if pool is not None:
        pool.shutdown()
    T["write_drain"] = time.perf_counter() - tc
    if elog is not None:
        elog.close()
    if wstat["error"] is not None:
        raise wstat["error"]
    n = n_done
    wall = time.perf_counter() - t_start
    summary = dict(events=n, wall_s=wall, events_per_s=n / max(wall, 1e-9),
                   per_event_s={k: v / max(n, 1) for k, v in T.items()},
                   kept_mean=float(np.mean(kept)) if kept else 0.0, device=str(device),
                   writer_thread_busy_s_per_event=wstat["busy"] / max(n, 1), workers=workers, deflate=deflate,
                   outputs=w.outputs, optical_compression={s: osamp[s] / max(okept[s], 1) for s in systems})
    if chain is not None:
        summary.update(config_digest=chain.digest, stages_per_event_s=None if chain.timings is None else
                       {k: v / max(n, 1) for k, v in chain.timings.items()})
    if calibrate is not None:
        table = normalization_table(cal)
        Path(calibrate).parent.mkdir(parents=True, exist_ok=True)
        np.save(calibrate, table)
        summary.update(calibrated=str(calibrate), norm_sigma_shape=list(table.shape))
    if cuda:
        summary.update(gpu_max_memory_gb=torch.cuda.max_memory_allocated() / 1e9, gpu=torch.cuda.get_device_name())
    summary.update(_storage(w.outputs, systems, n))
    return summary


def _storage(outputs, systems, n):
    import h5py

    def stored(path, group=None):
        tot = 0
        with h5py.File(path, "r") as f:
            for o in _datasets(f[group] if group else f):
                tot += o.id.get_storage_size()
        return tot
    tpc = [p for p in outputs if "_coeff_" in p and "_coeff_clean_" not in p]
    opt = [p for p in outputs if "_optical_" in p]
    out = {"tpc": sum(stored(p) for p in tpc) / 1e6 / max(n, 1)}
    for s in systems:
        out[s] = sum(stored(p, f"optical/{s}") for p in opt if s in h5py.File(p, "r")["optical"]) / 1e6 / max(n, 1)
    return dict(stored_mb_per_event=out, output_mb_per_event=sum(os.path.getsize(p) for p in outputs) / 1e6 / max(n, 1))


def _datasets(g):
    import h5py
    out = []
    g.visititems(lambda _, o: out.append(o) if isinstance(o, h5py.Dataset) else None)
    return out


DATASET = dict(sim="sim_wire", sbnd="sbnd", goop="goop")


def plan(source, out, run_dirs, events_per_task, events_per_shard):
    """A corpus's tasks, one line each: ``PREFIX FIRST COUNT EVENTS_PER_SHARD SHARD_START INPUT...``.
    sim: one task per sensor file, its shard number the file's (``..._sensor_0042.h5`` -> shard 42);
    sbnd / goop: every ``events_per_task`` events of a run's files."""
    lines = []
    for d in map(Path, run_dirs):
        prefix = f"{out}/{d.name}/{DATASET[source]}"
        if source == "sim":
            from helix.tpc.io import list_events
            for f in sorted(d.glob("*_sensor_*.h5")):
                lines.append(f"{prefix} 0 {len(list_events(str(f)))} {events_per_shard} {int(f.stem.rsplit('_', 1)[1])} {f}")
            continue
        files = sorted(str(f) for f in d.glob("*.h5"))
        n = len(make_source(source, files, None))
        for first in range(0, n, events_per_task):
            if first % events_per_shard:
                raise ValueError("events_per_task must be a multiple of events_per_shard")
            lines.append(f"{prefix} {first} {min(events_per_task, n - first)} {events_per_shard} "
                         f"{first // events_per_shard} {' '.join(files)}")
    return lines


def _complete(prefix, kind, shard_start, count, events_per_shard):
    """Every shard a task would write exists (shards appear only when complete)."""
    return all(Path(f"{prefix}_{kind}_{k:04d}.h5").exists()
               for k in range(shard_start, shard_start + -(-count // events_per_shard)))


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "plan":
        pp = argparse.ArgumentParser(prog="python -m helix.build plan", description=plan.__doc__)
        pp.add_argument("source", choices=SOURCES)
        pp.add_argument("out")
        pp.add_argument("run_dirs", nargs="+")
        pp.add_argument("--events-per-task", type=int, default=500)
        pp.add_argument("--events-per-shard", type=int, default=100)
        b = pp.parse_args(argv[1:])
        print("\n".join(plan(b.source, b.out, b.run_dirs, b.events_per_task, b.events_per_shard)))
        return None
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=SOURCES)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--output", required=True, help="DIR/PREFIX of the shards")
    ap.add_argument("--device", default=None)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--spread", type=int, default=None, help="N events evenly over all input events")
    ap.add_argument("--events-per-shard", type=int, default=100)
    ap.add_argument("--shard-start", type=int, default=0, help="number of the first shard")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--norm-sigma", default=None, help="the corpus-wide (G, n_bands) table (.npy) for every shard")
    g.add_argument("--calibrate", default=None, help="save the selected events' table here; write no shards")
    ap.add_argument("--config", default=None, help="TPC chain overrides (json / yaml)")
    ap.add_argument("--optical-config", default=None, help="per-system OpticalSpec overrides")
    ap.add_argument("--no-optical", action="store_true", help="TPC only")
    ap.add_argument("--workers", type=int, default=4, help="reader processes (sbnd)")
    ap.add_argument("--deflate", type=int, default=4, help="deflate level of the optical shards")
    ap.add_argument("--events-log", default=None, help="per-event jsonl (kept per plane / system, timing)")
    ap.add_argument("--profile", action="store_true", help="time the TPC chain's stages")
    ap.add_argument("--skip-complete", action="store_true",
                    help="do nothing when every shard of this selection exists (a resubmitted task)")
    a = ap.parse_args(argv)
    if a.skip_complete and a.count and a.calibrate is None and _complete(
            a.output, "optical" if a.source == "goop" else "coeff", a.shard_start, a.count, a.events_per_shard):
        print(f"complete: {a.output} shards from {a.shard_start}")
        return None

    from helix.tpc.chain import TPCChain
    src = make_source(a.source, a.input, a.device, optical=not a.no_optical)
    chain = ochain = None
    if hasattr(src, "chain_config"):
        chain = TPCChain(src.chain_config(_load_config(a.config)), device=a.device)
        if a.profile:
            chain.enable_timing()
    if src.systems and not a.no_optical and a.calibrate is None:
        ochain = src.optical_chain(_load_config(a.optical_config))
    norm = None if a.norm_sigma is None else np.load(a.norm_sigma).astype(np.float32)
    summary = build(src, a.output, indices=select(len(src), a.first, a.count, a.spread), chain=chain, ochain=ochain,
                    events_per_shard=a.events_per_shard, shard_start=a.shard_start, norm_sigma=norm,
                    calibrate=a.calibrate, workers=a.workers, deflate=a.deflate, events_log=a.events_log)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
