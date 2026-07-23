"""2x2 panels per (event, plane), FULL PLANE (not cropped):
clean | noisy(+coherent+intrinsic) | noise-removed (COEFFICIENT-SPACE smart removal
+ DWT recon) | diff (recon - clean).

Removal = the weight/coefficient-space level-aware gated common-mode remover
(research/coherent_coeffs/smart.py::smart_removal), ported here verbatim to avoid that
module's hardcoded import chain. NOT the sample-space helix.tpc.remove_coherent.

Style: SymLogNorm linthresh=2, RdBu_r. Data + noise via the GPU production pipeline.
"""
from __future__ import annotations
import sys, os
for p in ("/sdf/group/neutrino/omara/helix/.pylibs",
          "/sdf/group/neutrino/omara/particle-imaging-models/libs/pimm-data/src",
          "/sdf/group/neutrino/omara/helix"):
    if p not in sys.path:
        sys.path.insert(0, p)
import hdf5plugin  # noqa
import numpy as np, torch, pywt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

import measure_coeffs as M   # same folder
from pimm_data import JAXTPCDataset
from pimm_data.batch_transforms import move_to_device, _batch_seeds, BatchDensify, BatchAddIntrinsicNoise, BatchDigitize
from helix.core import backend
from helix.core.wavelet import sparsify, reconstruct
from helix.tpc.config import DetectorConfig

GS, WAVELET, LEVEL, MODE = 64, "coif3", 4, "periodization"
KGATE, KSIG = 4.0, 3.0   # kgate=4.0 matches every research fig/eval (smart_figs, oracle, de_*,
                         # induction); it minimizes leftover coherent. The smart.py *default* is
                         # 3.0 (max-F0 point) which leaves small coherent strips — the bug here.
EVENTS = [243, 46]
PLANES = [0, 1, 2]                      # volume_0 U, V, Y
OUTDIR = "/sdf/group/neutrino/omara/helix/research/coeff_foundation_model/figures"


# ── coefficient-space smart removal (ported from research/coherent_coeffs/smart.py) ──
def _block_common_mode(band, nw, ksig=KSIG):
    nblk = (nw + GS - 1) // GS
    L = band.shape[-1]
    Mc = np.zeros((nblk, L), np.float32)
    for g in range(nblk):
        lo, hi = g * GS, min((g + 1) * GS, nw)
        blk = band[lo:hi]
        med = np.median(blk, axis=0)
        resid = blk - med
        sg = max(np.median(np.abs(resid)) / 0.6745, 1e-6)
        uf = np.abs(resid) <= ksig * sg
        nuf = uf.sum(0)
        mean = (blk * uf).sum(0) / np.maximum(nuf, 1)
        Mc[g] = np.where(nuf > 0, mean, med)
    return Mc


def _broadcast_blocks(Mc, nw):
    nblk = Mc.shape[0]
    return Mc[np.minimum(np.arange(nw) // GS, nblk - 1)]


def smart_removal(noisy, kgate=KGATE, ksig=KSIG):
    """Coefficient-space level-aware gated common-mode removal -> (cleaned, coh_hat)."""
    nw, nt = noisy.shape
    bands = pywt.wavedec(noisy.astype(np.float32), WAVELET, level=LEVEL, mode=MODE, axis=-1)
    est = []
    for b in bands:
        Mc = _block_common_mode(b, nw, ksig)
        sigc = max(float(np.median(np.abs(Mc)) / 0.6745), 1e-6)     # robust coherent scale
        t = kgate * sigc
        gated = np.where(np.abs(Mc) < t, Mc, 0.0)                   # keep small=coherent, drop large=signal
        est.append(_broadcast_blocks(gated, nw))
    cleaned = pywt.waverec([b - e for b, e in zip(bands, est)], WAVELET, mode=MODE, axis=-1)[..., :nt]
    coh_hat = pywt.waverec(est, WAVELET, mode=MODE, axis=-1)[..., :nt]
    return cleaned.astype(np.float32), coh_hat.astype(np.float32)


def symlog_im(ax, img, title, vmax, cbar=True):
    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm)
    ax.set_title(title, fontsize=10); ax.set_xlabel("wire"); ax.set_ylabel("tick")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="ADC")
    return im


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    geom, nts = M.load_geom()
    cfg = DetectorConfig(num_time_steps=nts, group_size=GS)
    backend.set_backend("numpy")
    ds = JAXTPCDataset(data_root=M.DATA_ROOT, split=M.SPLIT,
                       modalities=("sensor",), dataset_name=M.DATASET_NAME)
    densify = BatchDensify(geom)
    addnoise = BatchAddIntrinsicNoise(geom, coherent=True, incoherent=True)
    digit = BatchDigitize(geom)

    for ev in EVENTS:
        b = move_to_device(M.build_batch(ds, ev, 1), "cuda")
        seeds = _batch_seeds(b, 0, 0, 0, 1)
        densify(b, seeds=seeds)
        clean = {g: v[0].clone() for g, v in b["sensor_dense"].items()}
        addnoise(b, seeds=seeds); digit(b, seeds=seeds)
        noisy = b["sensor_dense"]
        for g in PLANES:
            lab = geom[g]["label"]
            cl = clean[g].cpu().numpy()
            no = noisy[g][0].cpu().numpy()
            removed, coh_hat = smart_removal(no)                    # coeff-space coherent removal
            sp = sparsify(removed, wavelet=WAVELET, level=LEVEL, mode=MODE,
                          threshold=cfg.threshold_spec())           # then production denoise
            rc = reconstruct(sp, nts)
            diff = rc - cl
            sig = np.abs(cl) > 0
            f0 = 1.0 - np.abs(rc - cl)[sig].sum() / max(np.abs(cl)[sig].sum(), 1e-9)
            cohL = float(np.sqrt(np.mean(((coh_hat) - (no - cl))[~sig] ** 2)))  # coh est vs (noisy-clean) off-signal
            vmax = max(float(np.percentile(np.abs(cl[sig]), 99.5)) if sig.any() else 20.0, 30.0)
            fig, ax = plt.subplots(2, 2, figsize=(13, 9))
            symlog_im(ax[0, 0], cl, "clean (truth)", vmax)
            symlog_im(ax[0, 1], no, "noisy: +coherent +intrinsic", vmax)
            symlog_im(ax[1, 0], rc, "removed: coeff-space gate + DWT recon", vmax)
            symlog_im(ax[1, 1], diff, "diff (recon - clean)", max(vmax * 0.5, 20.0))
            fig.suptitle(f"event {ev}  {lab}  FULL PLANE ({cl.shape[0]}w x {cl.shape[1]}t)  "
                         f"coeff-space removal  F0={f0:.3f}", fontweight="bold")
            fig.tight_layout()
            out = f"{OUTDIR}/full_ev{ev}_{lab}.png"
            fig.savefig(out, dpi=120); plt.close(fig)
            print(f"saved {out}  vmax={vmax:.0f}  F0={f0:.3f}  coh_est_off_rms={cohL:.2f}")


if __name__ == "__main__":
    main()
