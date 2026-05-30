"""Total kept coefficients PER EVENT for several kappa (VisuShrink-hard, coif3 L10),
across 100 events. Saves big/kept_per_event.json. Usage: [n_events]"""
from __future__ import annotations
import sys, os, json
import numpy as np
import torch.multiprocessing as mp
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/scripts/optical")

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
OUT = "/sdf/group/neutrino/omara/helix/temp/figures/big"
WL, LEVEL = "coif3", 10
KAPPAS = [1.0, 1.2, 1.5, 1.75, 2.0, 2.5]


def worker(task):
    dev_id, events = task
    import torch
    torch.cuda.set_device(dev_id); dev = f"cuda:{dev_id}"
    from helix.core import backend; backend.set_backend("torch")
    from helix.optical import config_from_file
    from helix.optical.io import read_event_chunks, pad_batch
    from helix.core.wavelet_ops_torch import _wavedec
    from sweep_big import threshold, raw_sigma
    cfg0 = config_from_file(PATH)
    out = []
    for ev in events:
        ec = read_event_chunks(PATH, ev, cfg0)
        sig_np = raw_sigma(ec.chunks)
        bnp, lengths = pad_batch(ec.chunks, LEVEL)
        x = torch.as_tensor(bnp, device=dev); lens = torch.as_tensor(lengths, device=dev)
        L = x.shape[1]; valid = torch.arange(L, device=dev)[None, :] < lens[:, None]
        sig = torch.as_tensor(sig_np, device=dev).clamp_min(1e-9)
        co = _wavedec(x, WL, LEVEL)
        rec = dict(event=ev, samples=int(lengths.sum()), n_chunks=int(len(lengths)),
                   active=int(((x.abs() > 3 * sig[:, None]) & valid).sum()))
        for kap in KAPPAS:
            tc = threshold(co, sig, dict(method="visu", func="hard", kappa=kap), torch)
            rec[f"kept_{kap}"] = int(sum(int((c != 0).sum()) for c in tc))
        out.append(rec)
        print(f"[gpu{dev_id}] {ev}", flush=True)
    return out


def main():
    import torch
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    from helix.optical import list_events
    events = list_events(PATH)[:n]
    ng = torch.cuda.device_count()
    tasks = [(i, events[i::ng]) for i in range(ng) if events[i::ng]]
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        rows = [r for p in pool.map(worker, tasks) for r in p]
    json.dump({"kappas": KAPPAS, "rows": rows}, open(f"{OUT}/kept_per_event.json", "w"))

    samp = np.array([r["samples"] for r in rows])
    print(f"\nPer-event total coefficients ({len(rows)} events, coif3 L10, VisuShrink-hard)")
    print(f"raw samples/event: median {np.median(samp):,.0f}\n")
    print(f"{'kappa':>6s} {'mean':>9s} {'median':>9s} {'std':>8s} {'min':>8s} {'max':>9s} {'count-compr':>12s}")
    for kap in KAPPAS:
        k = np.array([r[f"kept_{kap}"] for r in rows])
        comp = (samp / k).mean()
        print(f"{kap:>6.2f} {k.mean():>9,.0f} {np.median(k):>9,.0f} {k.std():>8,.0f} "
              f"{k.min():>8,d} {k.max():>9,d} {comp:>11.1f}x")


if __name__ == "__main__":
    main()
