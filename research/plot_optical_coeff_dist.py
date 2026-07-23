"""Run optical metrics over real goop events at the default kappa=1.2 (1x-noise
operating point) and plot the distributions of KEPT wavelet coefficients.

Per stored chunk: DWT (coif3, L10, periodization); each detail band is hard-cut
at t = kappa * sigma * sqrt(2 ln N_band) with sigma = the chunk db1-MAD noise
(approx band kept untouched). Three figures (kept coefficients only):

  optical_kept_magnitude.png   distribution of kept |coefficient| (ADC), pooled.
  optical_coeff_perband.png    kept |coefficient| (ADC) per band + threshold.
  optical_kept_per_event.png   total kept coefficients per event (~166k).
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
import numpy as np
import pywt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from helix.optical import io as oio

LIGHT = "/sdf/home/y/youngsam/sw/dune/sim/goop/data/light_output.h5"
OUT = "/sdf/group/neutrino/omara/helix/temp/figures/current"
os.makedirs(OUT, exist_ok=True)
KAPPA = 1.2
WAV, LEVEL, MODE = "coif3", 10, "periodization"
N_EVENTS = int(os.environ.get("N_EVENTS", "25"))
NB = LEVEL + 1                                  # bands: A10 + D10..D1

cfg = oio.config_from_file(LIGHT)
events = oio.list_events(LIGHT)[:N_EVENTS]

# --- accumulators (KEPT detail coefficients only) ---
MBINS = np.logspace(0.7, 4.6, 110)              # |coeff| ADC bins (~5 .. 40000)
kept_hist = np.zeros(len(MBINS) - 1)            # faithful: every kept coeff counted
kept_mag = [[] for _ in range(NB)]              # subsample of kept |coeff| per band (for violins)
thr_band = [[] for _ in range(NB)]
tot_cnt = np.zeros(NB); kept_cnt = np.zeros(NB)
kept_per_event = []
CAP = 60000
rng = np.random.default_rng(0)

for ev in events:
    ec = oio.read_event_chunks(LIGHT, ev, cfg)
    sig = oio.chunk_noise_sigma(ec.chunks)
    ev_kept = 0
    for c, s in zip(ec.chunks, sig):
        x = np.asarray(c, np.float64)
        lev = min(LEVEL, pywt.dwt_max_level(len(x), pywt.Wavelet(WAV).dec_len))
        bands = pywt.wavedec(x, WAV, level=lev, mode=MODE)      # [cA, cD_lev,...,cD_1]
        ev_kept += len(bands[0])                                # approx kept untouched
        for j, b in enumerate(bands):
            if j == 0:
                continue                                        # approx: not thresholded
            bi = NB - (lev - (j - 1))                           # A->0, D10->1, ... D1->NB-1
            a = np.abs(b)
            t = KAPPA * float(s) * np.sqrt(2.0 * np.log(max(len(b), 2)))
            km = a[a >= t]
            ev_kept += km.size
            tot_cnt[bi] += a.size; kept_cnt[bi] += km.size; thr_band[bi].append(t)
            kept_hist += np.histogram(km, MBINS)[0]
            if km.size and len(kept_mag[bi]) < CAP:
                take = min(km.size, 3000)
                kept_mag[bi].extend(km[rng.integers(0, km.size, take)].tolist())
    kept_per_event.append(ev_kept)

kept_per_event = np.array(kept_per_event, float)
surv = kept_cnt / np.maximum(tot_cnt, 1)
band_names = ["A10"] + [f"D{LEVEL - k}" for k in range(LEVEL)]
print(f"{len(events)} events, κ={KAPPA}: mean {kept_per_event.mean():,.0f} kept coeffs/event "
      f"(±{kept_per_event.std():,.0f}); pooled-kept median |c| = "
      f"{np.exp(np.interp(0.5, np.cumsum(kept_hist)/kept_hist.sum(), np.log(np.sqrt(MBINS[1:]*MBINS[:-1])))):.0f} ADC")

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})

# ===== Fig 1: pooled distribution of kept |coefficient| =====
ctr = np.sqrt(MBINS[1:] * MBINS[:-1])
h = kept_hist / max(kept_hist.sum(), 1)
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.bar(ctr, h, width=np.diff(MBINS), color="#1f4e79", alpha=0.85, edgecolor="none", align="center")
tlo, thi = min(np.mean(t) for t in thr_band if t), max(np.mean(t) for t in thr_band if t)
ax.axvspan(MBINS[0], thi, color="0.5", alpha=0.10)
ax.axvline(thi, color="0.4", ls="--", lw=1.2)
ax.text(0.62, 0.92, f"all kept ≥ per-band threshold\nκσ√(2 ln N) ≈ {tlo:.0f}–{thi:.0f} ADC\n"
        f"median kept |c| ≈ 27 ADC", transform=ax.transAxes, fontsize=9.5, va="top", color="0.3",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.75", lw=0.6))
ax.set_xscale("log")
ax.set_xlabel("kept coefficient magnitude  |c|  [ADC]")
ax.set_ylabel("fraction of kept coefficients")
ax.set_title(f"Distribution of kept wavelet coefficients  ({len(events)} events, κ={KAPPA})",
             fontsize=12.5)
ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/optical_kept_magnitude.png", dpi=140)
print("wrote optical_kept_magnitude.png")

# ===== Fig 2: kept |coefficient| per band (ADC) + threshold =====
det = list(range(NB - 1, 0, -1))                # D1..D10 left->right
data = [np.array(kept_mag[o]) for o in det]
labs = [band_names[o] for o in det]
thr = [np.mean(thr_band[o]) if thr_band[o] else np.nan for o in det]
fig2, ax2 = plt.subplots(figsize=(10, 5.2))
xs = np.arange(len(det))
parts = ax2.violinplot([d if len(d) else [0] for d in data], positions=xs, widths=0.8, showextrema=False)
cmap = plt.cm.viridis(np.linspace(0.12, 0.9, len(det)))
for pc, c in zip(parts["bodies"], cmap):
    pc.set_facecolor(c); pc.set_alpha(0.7); pc.set_edgecolor("0.3")
ax2.plot(xs, thr, "D", ms=6, color="#c0392b", zorder=5)
ax2.set_yscale("log"); ax2.set_xticks(xs); ax2.set_xticklabels(labs)
ax2.set_xlabel("detail band   (finest D1 → coarsest D10)")
ax2.set_ylabel("kept |coefficient|  [ADC]")
ax2.set_title(f"Kept coefficient magnitudes per band  (κ={KAPPA})", fontsize=12.5, pad=24)
xa = ax2.get_xaxis_transform()
for x, s in zip(xs, [surv[o] for o in det]):
    ax2.text(x, 1.015, f"{s*100:.0f}%", transform=xa, ha="center", va="bottom",
             fontsize=8.5, color="0.3", fontweight="medium")
ax2.text(0.985, 0.04, "◆  threshold  κσ√(2 ln N)\ntop %  =  fraction of band kept",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=9, color="0.3",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", lw=0.6))
ax2.grid(axis="y", alpha=0.2); ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout(); fig2.savefig(f"{OUT}/optical_coeff_perband.png", dpi=140)
print("wrote optical_coeff_perband.png")

# ===== Fig 3: kept coefficients per event =====
fig3, ax3 = plt.subplots(figsize=(8.5, 4.6))
ax3.hist(kept_per_event / 1e3, bins=18, color="#1f4e79", alpha=0.85, edgecolor="white")
mu = kept_per_event.mean() / 1e3
ax3.axvline(mu, color="#c0392b", lw=1.8, label=f"mean {mu:.0f}k")
ax3.set_xlabel("kept coefficients per event  [×10³]"); ax3.set_ylabel("events")
ax3.set_title(f"Total kept coefficients per event at the 1×-noise point  (κ={KAPPA}, {len(events)} events)",
              fontsize=12)
ax3.legend(frameon=True); ax3.grid(axis="y", alpha=0.2)
ax3.spines[["top", "right"]].set_visible(False)
fig3.tight_layout(); fig3.savefig(f"{OUT}/optical_kept_per_event.png", dpi=140)
print("wrote optical_kept_per_event.png")
