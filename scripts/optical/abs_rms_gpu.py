"""Absolute-RMS (ADC) rate-distortion data, computed on GPU (multi-process),
saved to JSON for fast/iterable plotting. RMS = ||recon - denoised-ref(kappa=1)||
over active signal samples, event-total aggregation. Compression = event-total
fixed-bit. Usage: python scripts/optical/abs_rms_gpu.py [n_events]
"""
from __future__ import annotations
import sys, os, json, math
import numpy as np
import torch.multiprocessing as mp
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/scripts/optical")

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
OUT = "/sdf/group/neutrino/omara/helix/temp/figures/big"
WLS, LEVEL = ("coif3", "sym6"), 10
KAPPAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0]
BITS = [6, 8, 10, 12, 16]


def worker(task):
    dev_id, events = task
    import torch
    torch.cuda.set_device(dev_id); dev = f"cuda:{dev_id}"
    from helix.core import backend; backend.set_backend("torch")
    from helix.optical import config_from_file
    from helix.optical.io import read_event_chunks, pad_batch
    from helix.core.wavelet_ops_torch import _wavedec, _waverec
    from sweep_big import threshold, quant, raw_sigma
    cfg0 = config_from_file(PATH)
    acc = {}   # label -> [sqerr, count, raw_bits, storage_bits]
    for ev in events:
        ec = read_event_chunks(PATH, ev, cfg0)
        sig_np = raw_sigma(ec.chunks)
        bnp, lengths = pad_batch(ec.chunks, LEVEL)
        x = torch.as_tensor(bnp, device=dev); lens = torch.as_tensor(lengths, device=dev)
        L = x.shape[1]; valid = torch.arange(L, device=dev)[None, :] < lens[:, None]
        sig = torch.as_tensor(sig_np, device=dev).clamp_min(1e-9)
        active = (x.abs() > 3 * sig[:, None]) & valid
        raw_bits = float((lens.float() * 16).sum())
        for wl in WLS:
            co = _wavedec(x, wl, LEVEL)
            ref = _waverec(threshold(co, sig, dict(method="visu", func="hard", kappa=1.0), torch), wl)[:, :L]
            for kap in KAPPAS:
                th = threshold(co, sig, dict(method="visu", func="hard", kappa=kap), torch)
                for b in BITS:
                    q = quant(th, b, torch)
                    rec = _waverec(q, wl)[:, :L]
                    d = (rec - ref)[active]
                    sqerr = float((d * d).sum()); cnt = int(active.sum())
                    st = 0.0
                    for c in q:
                        Lb = c.shape[1]; K = (c != 0).sum(1).float()
                        idx = math.ceil(math.log2(max(Lb, 2)))
                        st += float(torch.where(K <= 0.5 * Lb, K * (b + idx), torch.full_like(K, float(Lb * b))).sum())
                    lab = f"{wl}|{kap}|{b}"
                    a = acc.setdefault(lab, [0.0, 0, 0.0, 0.0])
                    a[0] += sqerr; a[1] += cnt; a[2] += raw_bits; a[3] += st
        print(f"[gpu{dev_id}] {ev}", flush=True)
    return acc


def main():
    import torch
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    from helix.optical import list_events
    events = list_events(PATH)[:n]
    ng = torch.cuda.device_count()
    tasks = [(i, events[i::ng]) for i in range(ng) if events[i::ng]]
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        parts = pool.map(worker, tasks)
    acc = {}
    for p in parts:
        for lab, a in p.items():
            b = acc.setdefault(lab, [0.0, 0, 0.0, 0.0])
            for i in range(4):
                b[i] += a[i]
    rows = []
    for lab, a in acc.items():
        wl, kap, bits = lab.split("|")
        rows.append(dict(wavelet=wl, kappa=float(kap), bits=int(bits),
                         rms=math.sqrt(a[0] / max(a[1], 1)), comp=a[2] / max(a[3], 1)))
    json.dump(dict(noise=2.566, rows=rows), open(f"{OUT}/abs_rms.json", "w"))
    print(f"saved abs_rms.json ({len(rows)} configs, {n} events)")


if __name__ == "__main__":
    main()
