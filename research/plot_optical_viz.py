"""Optical visualizations on a real goop light event — three clean,
single-purpose figures (current helix.optical path; numpy = pywt-exact, same
coeffs as the torch GPU production):

  read stored chunks -> VisuShrink-hard DWT (coif3 L10, κ=1.2) -> quantize 12b
  -> reconstruct.

  optical_decomposition.png  signal + multi-scale DWT map (helix.optical.viz)
  optical_denoise_1d.png     a bright PMT pulse: raw vs reconstructed + residual
  optical_event_2d.png       one PMT side stitched to 2-D: raw vs reconstructed
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from helix.core.backend import set_backend
from helix.core.wavelet import sparsify, reconstruct
from helix.optical import io as oio
from helix.optical.config import OpticalConfig
from helix.optical import viz

set_backend("numpy")
LIGHT = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
OUT = "/sdf/group/neutrino/omara/helix/temp/figures/current"
os.makedirs(OUT, exist_ok=True)
EVENT = "event_000"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})

cfg = oio.config_from_file(LIGHT)        # real pedestal (~29490) -> chunks are pedestal-subtracted
ec = oio.read_event_chunks(LIGHT, EVENT, cfg)
batch, lengths = oio.pad_batch(ec.chunks, cfg.dwt_level)
sigma = oio.chunk_noise_sigma(ec.chunks)
res = sparsify(batch, wavelet=cfg.wavelet, level=cfg.dwt_level, mode=cfg.dwt_mode,
               threshold=cfg.threshold, sigma=sigma)
res.coeffs = oio.quantize_coeffs(res.coeffs, cfg.quant_bits)
recon = np.asarray(reconstruct(res, batch.shape[1]))

kept = np.zeros(len(ec.chunks), int)
for c in res.coeffs:
    kept += (np.asarray(c) != 0).sum(1)
comp = lengths / np.maximum(kept, 1)
peaks = np.array([np.abs(batch[i, :lengths[i]]).max() for i in range(len(lengths))])
unsat = peaks < 0.6 * cfg.pedestal            # exclude rail-clipped pulses (clearer shape)
bright = int(np.argmax(np.where(unsat, peaks, -1)))
print(f"{len(ec.chunks)} chunks; brightest unsaturated #{bright}: peak={peaks[bright]:.0f} ADC, "
      f"{comp[bright]:.0f}× compression ({kept[bright]} coeffs)")

xb = batch[bright, :lengths[bright]]
rb = recon[bright, :lengths[bright]]

# ===================== Fig 1: decomposition map (library viz) =====================
fig1 = plt.figure(figsize=(10, 8.5))
viz.plot_decomposition(xb, tick_ns=cfg.tick_ns, wavelet=cfg.wavelet, level=cfg.dwt_level,
                       mode=cfg.dwt_mode, style="map", fig=fig1,
                       title=f"Optical PMT pulse — multi-scale DWT decomposition  "
                             f"(PMT {ec.pmt_id[bright]} {ec.side[bright]}, {EVENT})")
fig1.savefig(f"{OUT}/optical_decomposition.png", dpi=140, bbox_inches="tight")
print(f"wrote optical_decomposition.png")

# ===================== Fig 2: 1-D denoise (one bright chunk) =====================
s0, s1 = viz.prompt_window(xb, cfg.tick_ns, before_ns=600, width_ns=9000)
t = np.arange(s0, s1) * cfg.tick_ns / 1000.0
fig2, (a0, a1) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                              gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.07})
a0.plot(t, xb[s0:s1], color="0.62", lw=1.3, label="raw")
a0.plot(t, rb[s0:s1], color="#1f4e79", lw=1.0, label="reconstructed")
a0.legend(loc="lower right", frameon=True)
a0.set_ylabel("ADC")
a0.set_title(f"Optical PMT pulse — VisuShrink-hard DWT (coif3, L{cfg.dwt_level}, κ={cfg.threshold.scale}) "
             f"+ {cfg.quant_bits}-bit quant   ·   {comp[bright]:.0f}× compression",
             fontsize=12, pad=6)
a1.plot(t, (rb - xb)[s0:s1], color="#c0392b", lw=0.7)
a1.axhspan(-sigma[bright], sigma[bright], color="0.80", alpha=0.6,
           label=f"±noise σ ({sigma[bright]:.1f} ADC)")
a1.legend(loc="upper right", frameon=True, fontsize=9.5)
a1.set_ylabel("residual"); a1.set_xlabel("time [µs]")
for ax in (a0, a1):
    ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
fig2.savefig(f"{OUT}/optical_denoise_1d.png", dpi=140, bbox_inches="tight")
print("wrote optical_denoise_1d.png")

# ===================== Fig 3: 2-D event display (raw vs reconstructed) =====================
side = ec.side[bright]
m = ec.side == side
t0 = ec.t0_ns[m].min()
start = np.round((ec.t0_ns[m] - t0) / cfg.tick_ns).astype(int)
pid = ec.pmt_id[m]; lm = lengths[m]; idx = np.nonzero(m)[0]
nbins = int((start + lm).max())
raw2d = np.zeros((cfg.n_pmts_per_side, nbins), np.float32)
rec2d = np.zeros_like(raw2d)
for k, gi in enumerate(idx):
    n = lm[k]
    raw2d[pid[k], start[k]:start[k] + n] = batch[gi, :n]
    rec2d[pid[k], start[k]:start[k] + n] = recon[gi, :n]
V = np.percentile(np.abs(raw2d[raw2d != 0]), 99.9)
norm = SymLogNorm(linthresh=max(3.0, float(sigma.mean())), vmin=-V, vmax=V, base=10)
tmax = nbins * cfg.tick_ns / 1000.0
fig3, ax3 = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"hspace": 0.14})
for ax, img, ttl in zip(ax3, [raw2d, rec2d], ["raw", "reconstructed"]):
    im = ax.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm,
                   extent=[0, tmax, 0, cfg.n_pmts_per_side], interpolation="nearest")
    ax.set_ylabel(f"{ttl}\nPMT")
ax3[-1].set_xlabel("time [µs]")
cb = fig3.colorbar(im, ax=list(ax3), fraction=0.018, pad=0.012); cb.set_label("ADC (symlog)")
ev_comp = lm.sum() / max(kept[m].sum(), 1)
fig3.suptitle(f"Optical event display — {side} side ({EVENT}):  raw vs reconstructed   "
              f"·   {int(m.sum())} PMTs, {ev_comp:.0f}× compression", fontsize=12.5, y=0.96)
fig3.savefig(f"{OUT}/optical_event_2d.png", dpi=140, bbox_inches="tight")
print("wrote optical_event_2d.png")
