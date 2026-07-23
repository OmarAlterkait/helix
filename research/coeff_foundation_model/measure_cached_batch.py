#!/usr/bin/env python
"""Test: with coefficients PRE-EXTRACTED (cached), how much can we batch per GPU
and how fast? MAE encoder-drop (encode ~50% visible tokens), flash-varlen pack,
sweep K events/batch until OOM. Reports max K, throughput, 10M training time.
"""
import sys, time
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/.pylibs")
sys.path.insert(0, "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src")
sys.path.insert(0, "/sdf/group/neutrino/omara/helix")
import hdf5plugin  # noqa
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
import baseline_tpc as bl, star_tpc as stp
from measure_batching import BatchedViT
from star_model import DEV
VIS = 0.5                      # MAE: encoder sees 50% visible tokens


def cache_events(n):
    P = stp._pipeline(); P["nw"] = np.array([P["geom"][g]["n_wires"] for g in range(6)], np.int64)
    print(f"pre-extracting {n} events (one-time)...", flush=True)
    evs = []
    for i in range(n):
        B = bl.event_batch(i)
        # keep only VISIBLE tokens (MAE encoder-drop) -> cache the visible cell-level tensors
        vis = torch.rand(B["n_cells"], device=DEV) < VIS
        idx = torch.nonzero(vis).squeeze(1)
        e = {k: B[k][idx] for k in ("inp", "occ", "cell_band", "cell_gid", "cell_t", "cell_wire")}
        e["n"] = len(idx)
        evs.append(e)
    return evs


def main(d=192, blocks=4):
    evs = cache_events(64)
    nt = [e["n"] for e in evs]
    print(f"visible tokens/event: mean {np.mean(nt):.0f} (={VIS:.0%} of ~30k)")
    m = BatchedViT(d=d, blocks=blocks).cuda(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def run(K, n=8):
        torch.cuda.reset_peak_memory_stats()
        batch = evs[:K]
        P = {k: torch.cat([e[k] for e in batch]) for k in
             ("inp", "occ", "cell_band", "cell_gid", "cell_t", "cell_wire")}
        cells = [e["n"] for e in batch]
        cu = torch.tensor(np.concatenate([[0], np.cumsum(cells)]), dtype=torch.int32, device=DEV)
        maxs = int(max(cells)); T = sum(cells)
        msk = torch.zeros(T, dtype=torch.bool, device=DEV)   # encoder sees all visible
        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ol, vp = m(P, msk, cu, maxs)
                loss = ol.float().mean() + vp.float().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        for _ in range(2): step()
        torch.cuda.synchronize(); s = time.time()
        for _ in range(n): step()
        torch.cuda.synchronize()
        return (time.time() - s) / n * 1000, T, torch.cuda.max_memory_allocated() / 1e6

    print(f"\nMAE encoder-drop ({VIS:.0%} visible), d={d} {blocks}blk, flash-varlen:")
    print(f"{'K(ev)':>6} {'enc-tok':>9} {'ms/step':>8} {'peak GB':>8} {'events/s':>9} {'tok/s':>10}")
    best = None
    for K in (1, 2, 4, 8, 16, 32, 64):
        try:
            ms, T, mem = run(K)
            evps = 1000 * K / ms
            print(f"{K:>6} {T:>9} {ms:>8.0f} {mem/1e3:>8.1f} {evps:>9.1f} {1000*T/ms:>10.0f}", flush=True)
            best = (K, evps)
        except RuntimeError as ex:
            print(f"{K:>6}: {'OOM' if 'memory' in str(ex).lower() else str(ex)[:40]}"); break
    # 10M training-time extrapolation
    K, evps = best
    print(f"\n10M-event training (this small model, cached tokens, 1 GPU @ {evps:.0f} ev/s):")
    for ep in (1, 3, 10):
        gh = 10e6 * ep / evps / 3600
        print(f"  {ep} epoch(s): {gh:>6.0f} GPU-h  |  wall 1GPU {gh/24:.1f} d  |  5GPU {gh/5/24:.1f} d")


if __name__ == "__main__":
    main()
