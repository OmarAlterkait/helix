"""goop light files through the optical codec into optical shards (the SBND optical format).

goop (HELIX's simulated PMT light: two sides of 81 PMTs, 1 ns ticks, white noise ~2.6 ADC on a
~29490 pedestal, 15-bit codes) runs through the same method as SBND's PMTs and X-ARAPUCAs
(:mod:`helix.optical.codec`, :data:`helix.optical.codec.GOOP`: deconvolution with goop's own
measured response, a Haar difference kept where the light or the departure from it reaches its
threshold, coarse sums always) and is written in the same shard format
(:func:`helix.tpc.sparse_io.read_optical`, :func:`helix.optical.tokens.shard_tokens`), so
simulation and data reach the model in one representation. Per event the stored chunks of both
sides are one batch; channel = pmt_id (+1000 on the west side). Extra in the file:
``truth/pe_counts`` (n_events, 2, 81: goop's photon count per PMT, east / west) and attrs
``goop_events`` (the light file's event keys) and ``summary_json``.

The coif3 pipeline (:func:`helix.optical.pipeline.process_event`) stays as the reference: it
loses most isolated photons and part of the slow light (SBND-data-helix scripts/goop_compare.py).

  python -m helix.optical.goop_run --input light_output.h5 --output goop_shard.h5 [--device cuda]
      [--first 0] [--count N]
"""
from __future__ import annotations

import argparse
import json
import time

import h5py
import numpy as np


def event_waveforms(f, key):
    """The stored chunks of one event, both sides: dict(channel, t_us, length, offset, buf) and
    pe_counts (2, n_pmt) east / west."""
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--deflate", type=int, default=4)
    a = ap.parse_args(argv)

    from helix.optical.codec import GOOP, OpticalChain
    from helix.tpc.chain import TPCChain
    from helix.tpc.sparse_io import Writer
    chain = OpticalChain(GOOP, device=a.device, templates="goop")
    sync = (lambda: __import__("torch").cuda.synchronize()) if a.device and a.device.startswith("cuda") else (lambda: None)
    f = h5py.File(a.input, "r")
    keys = sorted(k for k in f.keys() if k.startswith("event_"))
    keys = keys[a.first:a.first + a.count if a.count is not None else None]
    w = Writer(a.output, TPCChain(), level=a.deflate)                    # optical-only shard (TPC config attrs unused)
    pes, kept, samples, t_codec = [], 0, 0, 0.0
    t_start = time.perf_counter()
    for i, key in enumerate(keys):
        wf, pe = event_waveforms(f, key)
        t0 = time.perf_counter()
        res = chain.process("goop", wf)
        sync(); t_codec += time.perf_counter() - t0
        w.add_optical(res); w.flush()
        kept += res["n_kept"]; samples += res["n_samples"]; pes.append(pe)
        if i < 3 or (i + 1) % 20 == 0:
            print(f"{key}: {res['n_kept']} values / {res['n_samples']} samples", flush=True)
    w.close()
    wall = time.perf_counter() - t_start
    n = max(len(keys), 1)
    with h5py.File(a.output, "a") as o:
        m = max((p.shape[1] for p in pes), default=0)
        o.create_dataset("truth/pe_counts", data=np.stack([np.pad(p, ((0, 0), (0, m - p.shape[1]))) for p in pes])
                         if pes else np.zeros((0, 2, 0)))
        o.attrs["goop_events"] = json.dumps(keys)
        mb = sum(ds.id.get_storage_size() for ds in _datasets(o["optical/goop"])) / 1e6 if "optical/goop" in o else 0.0
        summary = dict(events=len(keys), values_per_event=kept / n, samples_per_event=samples / n,
                       compression=samples / max(kept, 1), stored_mb_per_event=mb / n, codec_s_per_event=t_codec / n,
                       events_per_s=len(keys) / wall, device=a.device or "numpy")
        o.attrs["summary_json"] = json.dumps(summary)
    print(json.dumps(summary, indent=1))


def _datasets(g):
    out = []
    g.visititems(lambda _, obj: out.append(obj) if isinstance(obj, h5py.Dataset) else None)
    return out


if __name__ == "__main__":
    main()
