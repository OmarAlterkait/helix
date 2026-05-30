"""Per-DWT-level coefficient-count distributions for the DEFAULT optical method.

For each stored chunk: coif3 / L10 / periodization DWT, then the production
helix.core universal-hard threshold (t = κ·σ·sqrt(2 ln N_band) per detail BAND,
padding-free per-chunk σ; the approximation band is kept whole). Counts the
surviving coefficients in every band ([A10, D10, …, D1]) for every chunk, so we
can see *where* in the multi-scale decomposition the kept coefficients live and
how that varies across chunks/events.

Multi-GPU, sharded across torch.cuda.device_count(). Saves big/coeffs_per_level.json
(per-event arrays — restyle plots without recomputing).
Usage: python scripts/optical/coeffs_per_level.py [n_events] [kappa]
"""
from __future__ import annotations
import sys, os, json, math
import numpy as np
import torch.multiprocessing as mp
sys.path.insert(0, "/sdf/group/neutrino/omara/helix/scripts/optical")

PATH = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
OUT = "/sdf/group/neutrino/omara/helix/temp/figures/big"
WL, LEVEL = "coif3", 10
KAPPA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2     # default = 1× noise-RMS
SIGNAL_PEAK_MIN = 50.0       # |x|max > 50 ADC marks a signal chunk (metrics convention)


def worker(task):
    dev_id, events, kappa = task
    import torch
    torch.cuda.set_device(dev_id); dev = f"cuda:{dev_id}"
    from helix.core import backend; backend.set_backend("torch")
    from helix.optical import config_from_file
    from helix.optical.io import read_event_chunks, pad_batch
    from helix.core.wavelet_ops_torch import _wavedec
    from sweep_big import raw_sigma
    cfg0 = config_from_file(PATH)
    out = []
    for ev in events:
        ec = read_event_chunks(PATH, ev, cfg0)
        sig_np = raw_sigma(ec.chunks)                       # padding-free per-chunk σ
        bnp, lengths = pad_batch(ec.chunks, LEVEL)
        x = torch.as_tensor(bnp, device=dev)
        sig = torch.as_tensor(sig_np, device=dev).clamp_min(1e-9)
        peak = x.abs().amax(1)                              # per-chunk |x|max
        co = _wavedec(x, WL, LEVEL)                         # [cA, cD_L, …, cD_1]
        lev = len(co) - 1                                   # actual level (>=1)
        kept, bandlen = [], []
        for j, c in enumerate(co):
            bandlen.append(int(c.shape[1]))
            if j == 0:                                     # approximation kept whole
                kept.append(torch.full((c.shape[0],), c.shape[1], device=dev, dtype=torch.long))
            else:                                          # per-band universal-hard
                t = (kappa * sig * math.sqrt(2 * math.log(max(c.shape[1], 2))))[:, None]
                kept.append((c.abs() >= t).sum(1))
        keptM = torch.stack(kept, 1)                        # (n_chunks, n_bands)
        out.append(dict(
            event=ev,
            names=[f"A{lev}"] + [f"D{lev - k}" for k in range(lev)],
            bandlen=bandlen,
            peak=[round(float(v), 2) for v in peak.cpu().numpy()],
            length=[int(v) for v in lengths],
            kept=keptM.cpu().numpy().astype(int).tolist(),  # n_chunks × n_bands
        ))
        print(f"[gpu{dev_id}] {ev}", flush=True)
    return out


def main():
    import torch
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    from helix.optical import list_events
    events = list_events(PATH)[:n]
    ng = torch.cuda.device_count()
    tasks = [(i, events[i::ng], KAPPA) for i in range(ng) if events[i::ng]]
    with mp.get_context("spawn").Pool(len(tasks)) as pool:
        rows = [r for p in pool.map(worker, tasks) for r in p]
    json.dump(dict(wavelet=WL, level=LEVEL, kappa=KAPPA,
                   signal_peak_min=SIGNAL_PEAK_MIN, rows=rows),
              open(f"{OUT}/coeffs_per_level.json", "w"))

    # ---- console summary (signal chunks only) ----
    names = rows[0]["names"]; nb = len(names)
    agg_kept = np.zeros(nb); agg_tot = np.zeros(nb); nsig = 0; nnoise = 0
    noise_kept = 0.0
    for r in rows:
        kept = np.array(r["kept"]); peak = np.array(r["peak"]); bl = np.array(r["bandlen"])
        mask = peak > SIGNAL_PEAK_MIN
        agg_kept += kept[mask].sum(0); agg_tot += bl * int(mask.sum()); nsig += int(mask.sum())
        nnoise += int((~mask).sum()); noise_kept += float(kept[~mask].sum())
    tot_kept = agg_kept.sum()
    print(f"\nDefault method: coif3 L10 universal-hard κ={KAPPA}  ({len(rows)} events)")
    print(f"signal chunks (|x|max>{SIGNAL_PEAK_MIN:.0f}): {nsig}    noise chunks: {nnoise} "
          f"(mean {noise_kept/max(nnoise,1):.1f} kept/chunk)")
    print(f"\n{'band':>5s} {'scale[ns]':>9s} {'bandlen':>9s} {'kept/sigchunk':>13s} {'frac kept':>10s} {'share':>7s}")
    lev = nb - 1
    for i, nm in enumerate(names):
        sc = (1 << lev) if i == 0 else (1 << (lev - i + 1))         # approx & detail widths in samples
        print(f"{nm:>5s} {sc:>9d} {agg_tot[i]/max(nsig,1):>9,.0f} {agg_kept[i]/max(nsig,1):>13,.1f} "
              f"{agg_kept[i]/max(agg_tot[i],1):>10.4f} {agg_kept[i]/max(tot_kept,1):>6.1%}")
    print(f"\ntotal kept/sig-chunk: {tot_kept/max(nsig,1):,.0f}")


if __name__ == "__main__":
    main()
