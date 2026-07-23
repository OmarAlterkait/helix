"""TPC rate-fidelity sweep on real doraemon events: F0, residual-noise RMS, and
compression vs the wavelet threshold κ, per plane type. Coherent removal (with
the true per-wire σ) is κ-independent, so it is run once per (event,plane) and
only the sparsify step is swept. One clean 3-panel figure.
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.dirname(_HERE))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from helix.tpc.config import DetectorConfig
from helix.tpc.coherent import remove_coherent
from helix.core.wavelet import sparsify, reconstruct, ThresholdSpec
import _tpc_common as C

OUT = "/sdf/group/neutrino/omara/helix/temp/figures/current"
os.makedirs(OUT, exist_ok=True)
EVENTS = [0, 1, 2, 3]
PLANES = {"Y (collection)": "volume_0_Y", "U (induction)": "volume_0_U", "V (induction)": "volume_0_V"}
KAPPAS = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
DEFAULT_K = 1.0


def sweep_plane(label, event):
    clean, nt, wl, ped = C.load_clean(event, label)
    noisy = C.make_noisy(clean, wl, ped, seed=1000 + event)
    cfg = DetectorConfig(group_size=C.GROUP_SIZE, num_time_steps=nt, plane_labels=(label,))
    cleaned = np.asarray(remove_coherent(noisy, cfg, sigma_per_wire=C.intrinsic_sigma(wl)))
    f0s, nrms, comps = [], [], []
    for k in KAPPAS:
        spec = ThresholdSpec(method="universal", func="hard", scale=float(k),
                             per_band_sigma=True, threshold_approx=True)
        sp = sparsify(cleaned, wavelet=cfg.wavelet, level=cfg.dwt_level, mode=cfg.dwt_mode,
                      threshold=spec)
        recon = np.asarray(reconstruct(sp, nt))[:, :nt]
        f0s.append(C.f0(clean, recon)); nrms.append(C.rms_off(clean, recon))
        comps.append(sp.compression)
    return np.array(f0s), np.array(nrms), np.array(comps)


agg = {p: {"f0": [], "nrms": [], "comp": []} for p in PLANES}
for name, label in PLANES.items():
    for e in EVENTS:
        f0s, nrms, comps = sweep_plane(label, e)
        agg[name]["f0"].append(f0s); agg[name]["nrms"].append(nrms); agg[name]["comp"].append(comps)
    print(f"{name}: F0@κ1={np.mean([a[1] for a in agg[name]['f0']]):.3f} "
          f"comp@κ1={np.mean([a[1] for a in agg[name]['comp']]):.0f}x")

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})
colors = {"Y (collection)": "#1f4e79", "U (induction)": "#c0392b", "V (induction)": "#e67e22"}
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
specs = [("f0", "charge fidelity  F0", False),
         ("nrms", "residual noise RMS  [ADC]", False),
         ("comp", "compression ratio", True)]
for ax, (key, ylabel, logy) in zip(axs, specs):
    for name in PLANES:
        y = np.mean(agg[name][key], axis=0)
        ax.plot(KAPPAS, y, "-o", ms=4.5, color=colors[name], label=name, lw=1.8)
    ax.axvline(DEFAULT_K, color="0.5", ls="--", lw=1.2)
    ax.set_xlabel("threshold κ"); ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
axs[0].text(DEFAULT_K, axs[0].get_ylim()[0], " default", color="0.4", fontsize=9, va="bottom")
axs[0].legend(loc="lower left", fontsize=9.5, frameon=True)
fig.suptitle("HELIX TPC rate–fidelity sweep — real doraemon events  "
             f"(mean over {len(EVENTS)} events; coherent removal + VisuShrink-hard)",
             fontsize=12.5, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/tpc_metrics_sweep.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT}/tpc_metrics_sweep.png")
