"""Final stage: recompute curated frontier configs with ENTROPY-CODED rate
(realistic arithmetic-coding size), computed fully on-GPU. Rate per band =
n_kept*H(value symbols) + (n*L)*H_binary(sparsity). 100 events / 5 GPUs.

Reads big/final_configs.json. Saves big/final_gpu*.jsonl.
Usage: python scripts/optical/final_frontier.py [n_events]
"""
from __future__ import annotations
import sys, os, json, time, math
import numpy as np
import torch.multiprocessing as mp

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
OUT = "/sdf/group/neutrino/omara/helix/temp/figures/big"
NOISE = 2.566
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/scripts/optical")


def quant_codes(coeffs, bits, torch):
    """Return (dequantized list, integer-code list)."""
    deq, codes = [], []
    for c in coeffs:
        if bits >= 32:
            deq.append(c); codes.append(None); continue
        amax = c.abs().amax(1, keepdim=True).clamp_min(1e-12)
        step = amax / ((1 << (bits - 1)) - 1)
        q = torch.round(c / step)
        deq.append(q * step); codes.append(q)
    return deq, codes


def _H(v, torch):
    v = v.long(); v = v - int(v.min())
    counts = torch.bincount(v).float()
    p = counts[counts > 0] / counts.sum()
    return float(-(p * torch.log2(p)).sum())


def fixed_bits(deq, bits, torch):
    """Event-total stored bits at fixed bit-depth (same sparse/dense model)."""
    b = 0.0
    for c in deq:
        tot = c.numel(); K = int((c != 0).sum()); idx = math.ceil(math.log2(max(c.shape[1], 2)))
        b += K * (bits + idx) if K <= 0.5 * tot else tot * bits
    return b


def entropy_bits(codes, shapes, torch):
    """On-GPU entropy rate (bits/event). Same sparse/dense model as the fixed-bit
    metric; only the per-value cost changes B -> H(symbols)."""
    bits = 0.0
    for q, (n, Lcol) in zip(codes, shapes):
        if q is None:
            continue
        tot = n * Lcol
        K = int((q != 0).sum())
        if K == 0:
            continue
        idx = math.ceil(math.log2(max(Lcol, 2)))
        if K <= 0.5 * tot:                              # sparse: K*(H_value + index)
            bits += K * (_H(q[q != 0], torch) + idx)
        else:                                            # dense: all symbols incl zeros, no index
            bits += tot * _H(q.flatten(), torch)
    return bits


def worker(task):
    dev_id, events = task
    import torch
    torch.cuda.set_device(dev_id); dev = f"cuda:{dev_id}"
    from helix.core import backend; backend.set_backend("torch")
    from helix.optical import config_from_file
    from helix.optical.io import read_event_chunks, pad_batch
    from helix.core.wavelet_ops_torch import _wavedec, _waverec
    from sweep_big import threshold, raw_sigma, REF_WL, REF_LEVEL, REF_KAPPA
    cfg0 = config_from_file(PATH)
    configs = json.load(open(f"{OUT}/final_configs.json"))
    fout = open(f"{OUT}/final_gpu{dev_id}.jsonl", "w")
    for ev in events:
        ec = read_event_chunks(PATH, ev, cfg0)
        bnp, lengths = pad_batch(ec.chunks, 12)
        x = torch.as_tensor(bnp, device=dev); lens = torch.as_tensor(lengths, device=dev)
        if x.shape[1] < 65536:
            x = torch.nn.functional.pad(x, (0, 65536 - x.shape[1]))
        L = x.shape[1]; valid = torch.arange(L, device=dev)[None, :] < lens[:, None]
        sigmask = x.abs().amax(1) > 10 * NOISE
        sig = torch.as_tensor(raw_sigma(ec.chunks), device=dev).clamp_min(1e-9)  # padding-free
        rc = _wavedec(x, REF_WL, REF_LEVEL)
        ref = _waverec(threshold(rc, sig, dict(method="visu", func="hard", kappa=REF_KAPPA), torch), REF_WL)[:, :L]
        raw_bits = float((lens.float() * 16).sum())
        cache = {}
        for cfg in configs:
            try:
                wl, lev = cfg["wavelet"], cfg["level"]
                if (wl, lev) not in cache:
                    cache[(wl, lev)] = _wavedec(x, wl, lev)
                coeffs = cache[(wl, lev)]
                tc = threshold(coeffs, sig, cfg, torch)
                deq, codes = quant_codes(tc, cfg["bits"], torch)
                rec = _waverec(deq, wl)[:, :L]
                rm = rec * valid
                peak = float(((rm.min(1).values - (x * valid).min(1).values).abs() /
                              (x * valid).min(1).values.abs().clamp_min(1e-9))[sigmask].mean())
                rms = float(((rm - ref * valid).pow(2).sum(1).sqrt() /
                             (ref * valid).pow(2).sum(1).sqrt().clamp_min(1e-9))[sigmask].mean())
                ent = entropy_bits(codes, [(c.shape[0], c.shape[1]) for c in tc], torch)
                fx = fixed_bits(deq, cfg["bits"], torch)
                fout.write(json.dumps(dict(event=ev, comp_entropy=raw_bits / max(ent, 1),
                                           comp_fixed=raw_bits / max(fx, 1),
                                           peak_err=peak, rms_vs_ref=rms, **cfg)) + "\n")
            except Exception as e:
                fout.write(json.dumps(dict(event=ev, error=str(e)[:90], **cfg)) + "\n")
        fout.flush()
        print(f"[gpu{dev_id}] {ev} done", flush=True)
    fout.close()
    return f"gpu{dev_id} done"


def main():
    import torch
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    from helix.optical import list_events
    events = list_events(PATH)[:n]
    ng = torch.cuda.device_count()
    tasks = [(i, events[i::ng]) for i in range(ng) if events[i::ng]]
    print(f"FINAL: {len(events)} events x {len(json.load(open(f'{OUT}/final_configs.json')))} configs", flush=True)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        for r in pool.map(worker, tasks):
            print(r, flush=True)
    print(f"FINAL done in {time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
