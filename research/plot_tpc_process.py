"""TPC wire-plane progressive denoising on a real doraemon event — one clean
4-panel figure per plane (matches the existing temp/plots/out/plot_07_tpc2d.png):

  noisy  ->  coherent removed  ->  wavelet reconstruction  ->  truth

Current post-refactor path: pimm_data forward noise (coherent+incoherent) +
helix remove_coherent (given the true per-wire sigma) + sparsify/reconstruct.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent
from helix.core.wavelet import sparsify, reconstruct
import _tpc_common as C

OUT = "/sdf/group/neutrino/omara/helix/temp/figures/current"
os.makedirs(OUT, exist_ok=True)
EVENT = 0
WW, TW = 220, 760                       # crop: wires x ticks
PLANES = [("volume_0_Y", "collection (Y)"), ("volume_0_U", "induction (U)")]


def run(label, seed):
    clean, nt, wl, ped = C.load_clean(EVENT, label)
    noisy = C.make_noisy(clean, wl, ped, seed)
    cfg = DetectorConfig(group_size=C.GROUP_SIZE, num_time_steps=nt, plane_labels=(label,))
    cleaned = np.asarray(remove_coherent(noisy, cfg, sigma_per_wire=C.intrinsic_sigma(wl)))
    sp = sparsify(cleaned, wavelet=cfg.wavelet, level=cfg.dwt_level,
                  mode=cfg.dwt_mode, threshold=cfg.threshold_spec())
    recon = np.asarray(reconstruct(sp, nt))[:, :nt]
    m = dict(F0=C.f0(clean, recon), n_in=C.rms_off(clean, noisy),
             n_out=C.rms_off(clean, recon), comp=sp.compression)
    m["rej"] = m["n_in"] / max(m["n_out"], 1e-9)
    return clean, noisy, cleaned, recon, m, nt


def crop(clean, nt):
    ce = np.convolve(np.abs(clean).sum(0), np.ones(TW), "same")
    re = np.convolve(np.abs(clean).sum(1), np.ones(WW), "same")
    tc = int(np.clip(np.argmax(ce) - TW // 2, 0, nt - TW))
    wc = int(np.clip(np.argmax(re) - WW // 2, 0, clean.shape[0] - WW))
    return wc, tc


plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})
titles = ["noisy", "coherent removed", "wavelet reconstruction", "truth"]

for label, nice in PLANES:
    clean, noisy, cleaned, recon, m, nt = run(label, seed=1000)
    wc, tc = crop(clean, nt)
    sl = (slice(wc, wc + WW), slice(tc, tc + TW))
    panels = [noisy[sl], cleaned[sl], recon[sl], clean[sl]]
    V = max(40.0, np.percentile(np.abs(clean[sl]), 99.8))
    norm = SymLogNorm(linthresh=8.0, vmin=-V, vmax=V, base=10)

    fig, axs = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, img, t in zip(axs, panels, titles):
        im = ax.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm,
                       extent=[tc, tc + TW, wc, wc + WW], interpolation="nearest")
        ax.set_title(t, fontsize=12, pad=5)
        ax.set_xlabel("time tick")
    axs[0].set_ylabel("wire")
    cb = fig.colorbar(im, ax=axs, fraction=0.020, pad=0.012)
    cb.set_label("ADC (symlog)")
    fig.suptitle(f"HELIX TPC wire plane — progressive denoising   ({nice}, event {EVENT})",
                 fontsize=13.5, x=0.43, y=1.02)
    fig.text(0.43, -0.02,
             f"F0 = {m['F0']:.3f}     noise RMS {m['n_in']:.2f} → {m['n_out']:.2f} ADC "
             f"({m['rej']:.1f}× coherent+wavelet rejection)     compression {m['comp']:.0f}×",
             ha="center", fontsize=10.5, color="0.25")
    fn = f"{OUT}/tpc_denoise_{label.split('_')[-1]}.png"
    fig.savefig(fn, dpi=140, bbox_inches="tight")
    print(f"{label}: F0={m['F0']:.3f} noise {m['n_in']:.2f}->{m['n_out']:.2f} "
          f"rej={m['rej']:.1f}x comp={m['comp']:.0f}x -> {os.path.basename(fn)}")
