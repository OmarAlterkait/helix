"""2x2 panels per (event, plane): clean | noisy(+coherent+intrinsic) | noise-removed
(production remove_coherent -> sparsify -> reconstruct) | diff (recon - clean).

Style matches research/wire_denoise/viz_2x2.py (SymLogNorm linthresh=2, RdBu_r,
64-wire group lines). Data + noise via the GPU production pipeline; removal via the
helix.tpc production path (numpy backend) with the true per-wire intrinsic sigma.
"""
from __future__ import annotations
import sys
for p in ("/sdf/group/neutrino/omara/helix/.pylibs",
          "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src",
          "/sdf/group/neutrino/omara/helix"):
    if p not in sys.path:
        sys.path.insert(0, p)
import hdf5plugin  # noqa
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

import measure_coeffs as M   # same folder
from pimm_data import JAXTPCDataset
from pimm_data.batch_transforms import move_to_device, _batch_seeds, BatchDensify, BatchAddIntrinsicNoise, BatchDigitize
from helix.core import backend
from helix.tpc.config import DetectorConfig
from helix.tpc.pipeline import process_plane

GS = 64
EVENTS = [243, 46]                       # typical, busy
PLANES = [0, 1, 2]                       # volume_0: U, V, Y
OUTDIR = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/figures"


def best_crop(clean, hw=96, ht=240):
    en = np.abs(clean); nw, T = en.shape
    wc = en.sum(1); tc = en.sum(0)
    wi = max(0, min(int(np.argmax(np.convolve(wc, np.ones(hw), "same"))) - hw // 2, nw - hw))
    ti = max(0, min(int(np.argmax(np.convolve(tc, np.ones(ht), "same"))) - ht // 2, T - ht))
    return slice(wi, wi + hw), slice(ti, ti + ht)


def symlog_im(ax, img, title, ws, ts, vmax, cbar=True):
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    extent = [ws.start, ws.stop, ts.start, ts.stop]
    im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm, extent=extent)
    ax.set_title(title, fontsize=10); ax.set_xlabel("wire"); ax.set_ylabel("tick")
    first = ((ws.start // GS) + 1) * GS
    for g in range(first, ws.stop, GS):
        ax.axvline(g, color="gray", lw=0.5, ls="--", alpha=0.5)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="ADC")
    return im


def main():
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    geom, nts = M.load_geom()
    cfg = DetectorConfig(num_time_steps=nts, group_size=GS)   # production defaults (coif3 L4, kappa=1)
    backend.set_backend("numpy")                              # remove_coherent + sparsify (production)
    ds = JAXTPCDataset(data_root=M.DATA_ROOT, split=M.SPLIT,
                       modalities=("sensor",), dataset_name=M.DATASET_NAME)
    densify = BatchDensify(geom)
    addnoise = BatchAddIntrinsicNoise(geom, coherent=True, incoherent=True)
    digit = BatchDigitize(geom)

    for ev in EVENTS:
        b = move_to_device(M.build_batch(ds, ev, 1), "cuda")
        seeds = _batch_seeds(b, 0, 0, 0, 1)
        densify(b, seeds=seeds)
        clean = {g: v[0].clone() for g, v in b["sensor_dense"].items()}   # clean BEFORE noise
        addnoise(b, seeds=seeds); digit(b, seeds=seeds)
        noisy = b["sensor_dense"]

        for g in PLANES:
            lab = geom[g]["label"]
            cl = clean[g].cpu().numpy()
            no = noisy[g][0].cpu().numpy()
            sigma = cfg.wire_sigma_intrinsic(geom[g]["wire_lengths"])     # true per-wire intrinsic sigma
            rc = process_plane(no, cfg, sigma).reconstructed
            ws, ts = best_crop(cl)
            clc, noc, rcc = cl[ws, ts], no[ws, ts], rc[ws, ts]
            diff = rcc - clc
            vmax = max(float(np.abs(clc).max()), 20.0)
            fig, ax = plt.subplots(2, 2, figsize=(11, 8))
            symlog_im(ax[0, 0], clc, "clean (truth)", ws, ts, vmax)
            symlog_im(ax[0, 1], noc, "noisy: +coherent +intrinsic", ws, ts, vmax)
            symlog_im(ax[1, 0], rcc, "noise removed: remove_coherent + DWT recon", ws, ts, vmax)
            symlog_im(ax[1, 1], diff, "diff (recon - clean)", ws, ts, max(vmax * 0.5, 20.0))
            fig.suptitle(f"event {ev}  {lab}  (symlog ADC)  production pipeline", fontweight="bold")
            fig.tight_layout()
            out = f"{OUTDIR}/panel_ev{ev}_{lab}.png"
            fig.savefig(out, dpi=110); plt.close(fig)
            f0 = 1.0 - np.abs(rc - cl)[np.abs(cl) > 0].sum() / max(np.abs(cl)[np.abs(cl) > 0].sum(), 1e-9)
            print(f"saved {out}  vmax={vmax:.0f}  crop w{ws.start}-{ws.stop} t{ts.start}-{ts.stop}  F0={f0:.3f}")


if __name__ == "__main__":
    main()
