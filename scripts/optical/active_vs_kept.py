"""Per-chunk (active signal pixels) vs (kept coefficients) across all events, at
the three operating points. Saves big/active_vs_kept.json for reuse.
active pixels = #samples with |x|>3 sigma (per chunk). kept = #nonzero coeffs.
Usage: python scripts/optical/active_vs_kept.py [n_events]
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import torch.multiprocessing as mp
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/scripts/optical")

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
OUT = "/sdf/group/neutrino/omara/helix/temp/figures/big"
LEVEL = 10
# (label, noise-multiple, compression, wavelet, kappa)
OPS = [("1x", 1, 33, "coif3", 1.2), ("2x", 2, 49, "sym6", 1.75), ("3x", 3, 85, "coif3", 2.5)]


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
    out = {lab: {"active": [], "kept": []} for lab, *_ in OPS}
    for ev in events:
        ec = read_event_chunks(PATH, ev, cfg0)
        sig_np = raw_sigma(ec.chunks)
        bnp, lengths = pad_batch(ec.chunks, LEVEL)
        x = torch.as_tensor(bnp, device=dev); lens = torch.as_tensor(lengths, device=dev)
        L = x.shape[1]; valid = torch.arange(L, device=dev)[None, :] < lens[:, None]
        sig = torch.as_tensor(sig_np, device=dev).clamp_min(1e-9)
        active = ((x.abs() > 3 * sig[:, None]) & valid).sum(1).cpu().numpy()
        co_cache = {}
        for lab, m, comp, wl, kap in OPS:
            if wl not in co_cache:
                co_cache[wl] = _wavedec(x, wl, LEVEL)
            tc = threshold(co_cache[wl], sig, dict(method="visu", func="hard", kappa=kap), torch)
            kept = sum((c != 0).sum(1) for c in tc).cpu().numpy()
            out[lab]["active"].append(active); out[lab]["kept"].append(kept)
        print(f"[gpu{dev_id}] {ev}", flush=True)
    for lab in out:
        out[lab]["active"] = np.concatenate(out[lab]["active"]).tolist()
        out[lab]["kept"] = np.concatenate(out[lab]["kept"]).tolist()
    return out


def main():
    import torch
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    from helix.optical import list_events
    events = list_events(PATH)[:n]
    ng = torch.cuda.device_count()
    tasks = [(i, events[i::ng]) for i in range(ng) if events[i::ng]]
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        parts = pool.map(worker, tasks)
    merged = {lab: {"active": [], "kept": []} for lab, *_ in OPS}
    for p in parts:
        for lab in p:
            merged[lab]["active"] += p[lab]["active"]; merged[lab]["kept"] += p[lab]["kept"]
    json.dump({"ops": OPS, "data": merged}, open(f"{OUT}/active_vs_kept.json", "w"))
    print(f"saved active_vs_kept.json ({len(merged[OPS[0][0]]['active'])} chunks, {n} events)")


if __name__ == "__main__":
    main()
