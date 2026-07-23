"""Per-plane pipeline visualization in the fig10 style, self-contained on real data.

For a chosen event, per plane (U, V, Y), two rows:
  TOP  (image, symlog ADC): clean truth | noisy (+coherent+intrinsic) |
        after coherent removal (smart gate) | after wavelet sparsify (recon) |
        c - a  (clean - after-wavelet = total signal lost through the pipeline)
  BOT  (A4 coefficient map): noisy A4 (block stripes) | gated coherent estimate |
        after removal (c - a in coeff space, stripes gone)

Rebuilt from research/coherent_coeffs/smart_figs.py::fig_panel (which uses the
dead cc_common path) onto helix.tpc.io real sensor data + GPU-injected noise +
the canonical smart gate. kgate default = 3.0 (the qualified value).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/sdf/group/neutrino/omara/pimm-data/src")
sys.path.insert(0, os.path.join(HERE, "..", "coeff_foundation_model"))

import numpy as np
import pywt
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from helix.core.wavelet_ops_torch import _wavedec, _waverec
from helix.tpc.io import config_from_file, read_sensor_plane
from pimm_data.dense_ops import _coherent_torch, _incoherent_torch
from pimm_data.noise import (DEFAULT_ENC, DEFAULT_COH_RMS_ADC, DEFAULT_COH_CORNER_FREQ_HZ,
                             DEFAULT_COH_SLOPE, DEFAULT_COH_BETA, DEFAULT_SAMPLING_RATE_HZ, digitize)
from pimm_data.geometry import load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id
import measure_coeffs as MC

SHARD = ("/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor/"
         "run_0027575715/sim_wire_sensor_0000.h5")
NPZ = "/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz"
PLANES = ("volume_0_U", "volume_0_V", "volume_0_Y")
WAVELET, LEVEL, KAPPA, GS, DEV = "coif3", 4, 1.0, 64, "cuda"


def smart_clean(noisy_t, kgate):
    nt = noisy_t.shape[-1]
    pad = (-nt) % (1 << LEVEL)
    x = torch.nn.functional.pad(noisy_t, (0, pad)) if pad else noisy_t
    gated = MC.smart_gate_bands(_wavedec(x, WAVELET, LEVEL), kgate=kgate)
    return _waverec(gated, WAVELET)[..., :nt].cpu().numpy()


def wavelet_sparsify(img):
    """coif3 L4 per-band-MAD kappa=1 threshold -> reconstruction (pywt, the production basis)."""
    bands = pywt.wavedec(img.astype(np.float32), WAVELET, level=LEVEL, mode="periodization", axis=-1)
    thr = []
    for b in bands:
        sg = float(np.median(np.abs(b)) / 0.6745)
        t = KAPPA * sg * np.sqrt(2.0 * np.log(max(b.shape[-1], 2)))
        thr.append(np.where(np.abs(b) >= t, b, 0.0))
    return pywt.waverec(thr, WAVELET, mode="periodization", axis=-1)[..., :img.shape[-1]]


def _syml(ax, img, title, vmax, blocks=0):
    im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   norm=SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10))
    for g in range(1, blocks):
        ax.axvline(g * GS, color="k", lw=0.3, alpha=0.4)
    ax.set_title(title, fontsize=9); ax.set_xlabel("wire"); ax.set_ylabel("tick")
    return im


def panel(ev, label, cfg, reg, kgate=3.0):
    pt = label.split("_")[-1]
    ped = cfg.pedestals[pt]
    clean = read_sensor_plane(SHARD, ev, label, cfg.num_time_steps, ped)
    nw, nt = clean.shape
    wl = np.asarray(reg.get(canonical_plane_id(label), {}).get("wire_lengths", []), np.float64)
    wl = wl if len(wl) == nw else np.full(nw, 2.33)
    npz = np.load(NPZ, allow_pickle=True)
    gen = torch.Generator(device=DEV); gen.manual_seed(hash((ev, pt)) & 0xFFFFFFFF)
    coh = _coherent_torch(nw, nt, gen=gen, group_size=GS, rms_adc=DEFAULT_COH_RMS_ADC,
                          corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ, spectral_slope=DEFAULT_COH_SLOPE,
                          beta=DEFAULT_COH_BETA, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, device=DEV).cpu().numpy()
    inc = _incoherent_torch((nw, nt), torch.as_tensor(wl, dtype=torch.float32, device=DEV), gen=gen,
                            enc=DEFAULT_ENC, series_spectrum=(npz["spectrum_freqs_hz"], npz["spectrum_shape"]),
                            sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ, device=DEV).cpu().numpy()
    noisy = digitize(clean + coh + inc, ped)
    noisy_t = torch.as_tensor(noisy, device=DEV)
    cleaned = smart_clean(noisy_t, kgate)
    after_w = wavelet_sparsify(cleaned)                       # cleaned -> wavelet sparsify -> recon

    # signal-rich crop, SNAPPED to a 64-wire group boundary so the drawn group
    # lines coincide with the true coherent-group boundaries (else the block
    # stripes appear shifted from the lines — a display artifact, not a
    # removal bug: the removal always groups from wire 0 on the full image).
    e = np.abs(clean).sum(1)
    wc = int(np.argmax(np.convolve(e, np.ones(5 * GS), "same")))
    w0 = max(0, (wc // GS - 2) * GS); w1 = min(nw, w0 + 5 * GS); ws = slice(w0, w1)
    assert w0 % GS == 0
    tc = int(np.argmax(np.abs(clean[ws]).sum(0)))
    t0 = max(0, tc - 350); t1 = min(nt, t0 + 700); ts = slice(t0, t1)
    nblk = (w1 - w0) // GS + 1
    vmax = max(float(np.abs(clean[ws, ts]).max()), 30.0)

    fig, ax = plt.subplots(2, 5, figsize=(22, 8))
    _syml(ax[0, 0], clean[ws, ts], "true (clean signal)", vmax, nblk)
    _syml(ax[0, 1], noisy[ws, ts], "noisy: +coherent +intrinsic", vmax, nblk)
    _syml(ax[0, 2], cleaned[ws, ts], f"after coherent removal (smart k={kgate})", vmax, nblk)
    _syml(ax[0, 3], after_w[ws, ts], "after wavelet sparsify (recon)", vmax, nblk)
    im = _syml(ax[0, 4], (clean - after_w)[ws, ts], "c - a  (clean - after; signal lost)",
               max(vmax * 0.4, 15))
    fig.colorbar(im, ax=ax[0, 4], fraction=0.046)

    # A4 coeff row (bottom): noisy | gated coherent estimate | after removal (c-a)
    coh_hat = noisy - cleaned
    bn = pywt.wavedec(noisy, WAVELET, level=LEVEL, mode="periodization", axis=-1)[0][ws]
    be = pywt.wavedec(coh_hat, WAVELET, level=LEVEL, mode="periodization", axis=-1)[0][ws]
    ba = bn - be
    vc = max(float(np.percentile(np.abs(bn), 99.5)), 5.0)
    nrm = SymLogNorm(linthresh=max(vc * 0.05, 1.0), vmin=-vc, vmax=vc, base=10)
    for kk, (band, ttl) in enumerate([(bn, "A4 coeffs: noisy (block stripes)"),
                                      (be, "A4: gated coherent estimate"),
                                      (ba, "A4: after removal  c - a (stripes gone)")]):
        im2 = ax[1, kk].imshow(band.T, aspect="auto", origin="lower", cmap="RdBu_r", norm=nrm)
        for g in range(1, nblk):
            ax[1, kk].axvline(g * GS, color="k", lw=0.3, alpha=0.4)
        ax[1, kk].set_title(ttl, fontsize=9); ax[1, kk].set_xlabel("wire"); ax[1, kk].set_ylabel("A4 pos")
    fig.colorbar(im2, ax=ax[1, 2], fraction=0.046)
    # F0 summary
    sig = np.abs(clean) > 0
    tc_ = np.abs(clean)[sig].sum()
    f0_rem = 1 - np.abs(cleaned - clean)[sig].sum() / tc_
    f0_wav = 1 - np.abs(after_w - clean)[sig].sum() / tc_
    nkept = sum(int(np.count_nonzero(np.where(np.abs(b) >= KAPPA * np.median(np.abs(b)) / 0.6745
                * np.sqrt(2 * np.log(max(b.shape[-1], 2))), b, 0)))
                for b in pywt.wavedec(cleaned, WAVELET, level=LEVEL, mode="periodization", axis=-1))
    ax[1, 3].axis("off")
    ax[1, 3].text(0.0, 0.5, f"{pt} plane, event {ev}\n\nF0 after removal = {f0_rem:.4f}\n"
                  f"F0 after wavelet  = {f0_wav:.4f}\nkept coeffs = {nkept:,}\n\n"
                  f"gate keeps |m| < k·sigma_coh\n(small dense = coherent),\ndrops large (sparse = signal)",
                  fontsize=11, va="center")
    ax[1, 4].axis("off")
    fig.suptitle(f"{pt} plane — full pipeline: true -> +noise -> coherent removal (smart k={kgate}) "
                 f"-> wavelet sparsify   (symlog ADC)", fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, f"pipeline_{pt}_ev{ev}.png")
    fig.savefig(out, dpi=115); plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, nargs="+", default=[5, 40])
    ap.add_argument("--kgate", type=float, default=3.0)
    args = ap.parse_args()
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    cfg = config_from_file(SHARD)
    for ev in args.events:
        for label in PLANES:
            panel(ev, label, cfg, reg, args.kgate)


if __name__ == "__main__":
    main()
