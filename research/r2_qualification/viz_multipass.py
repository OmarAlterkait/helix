"""Visualize what the 2nd smart-gate pass does: 1-pass vs 2-pass, per plane.

Per plane, group-aligned crop, symlog ADC:
  row: noisy | after 1 pass | after 2 passes | (1p - 2p) = coherent the 2nd pass
       additionally removed (should be faint 64-wire block-stripes, group-aligned)
plus the off-signal residual-stripe RMS + kept-coeff count for each pass.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, "/sdf/group/neutrino/omara/pimm-data/src")
sys.path.insert(0, os.path.join(HERE, "..", "coeff_foundation_model"))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from helix.tpc.io import config_from_file, read_sensor_plane
from pimm_data.dense_ops import _coherent_torch, _incoherent_torch
from pimm_data.noise import digitize
from pimm_data.geometry import load_plane_registry
from pimm_data.jaxtpc import canonical_plane_id
import multipass as MP

GS = MP.GS


def _syml(ax, img, title, vmax, nblk=0):
    im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   norm=SymLogNorm(linthresh=1.0, vmin=-vmax, vmax=vmax, base=10))
    for g in range(1, nblk):
        ax.axvline(g * GS, color="k", lw=0.3, alpha=0.4)
    ax.set_title(title, fontsize=9); ax.set_xlabel("wire"); ax.set_ylabel("tick")
    return im


def panel(ev, pl, cfg, reg, npz):
    lab = f"volume_0_{pl}"; ped = cfg.pedestals[pl]
    clean = read_sensor_plane(MP.SHARD, ev, lab, cfg.num_time_steps, ped)
    nw, nt = clean.shape
    v = np.asarray(reg.get(canonical_plane_id(lab), {}).get("wire_lengths", []), np.float64)
    wlt = torch.as_tensor(v if len(v) == nw else np.full(nw, 2.33), dtype=torch.float32, device="cuda")
    gen = torch.Generator(device="cuda"); gen.manual_seed(hash((ev, pl)) & 0xFFFFFFFF)
    coh = _coherent_torch(nw, nt, gen=gen, group_size=GS, rms_adc=2.5, corner_freq_hz=20000.,
                          spectral_slope=1.5, beta=0.15, sampling_rate_hz=2e6, device="cuda")
    inc = _incoherent_torch((nw, nt), wlt, gen=gen, enc=(0.9, 0.79, 0.22),
                            series_spectrum=(npz["spectrum_freqs_hz"], npz["spectrum_shape"]),
                            sampling_rate_hz=2e6, device="cuda")
    noisy = digitize((torch.as_tensor(clean, device="cuda") + coh + inc).cpu().numpy(), ped)
    noisy_t = torch.as_tensor(noisy, device="cuda")
    c1 = MP.gate_img(noisy_t, 3.0, 1); c2 = MP.gate_img(noisy_t, 3.0, 2)
    cleanT = torch.as_tensor(clean, device="cuda")
    m1 = MP.metrics(c1, cleanT, coh, noisy_t); m2 = MP.metrics(c2, cleanT, coh, noisy_t)
    k1, k2 = MP.n_kept(c1), MP.n_kept(c2)
    c1, c2 = c1.cpu().numpy(), c2.cpu().numpy()

    e = np.abs(clean).sum(1); wc = int(np.argmax(np.convolve(e, np.ones(5 * GS), "same")))
    w0 = max(0, (wc // GS - 2) * GS); w1 = min(nw, w0 + 5 * GS); ws = slice(w0, w1)
    tcen = int(np.argmax(np.abs(clean[ws]).sum(0))); t0 = max(0, tcen - 350); t1 = min(nt, t0 + 700); ts = slice(t0, t1)
    nblk = (w1 - w0) // GS + 1
    vmax = max(float(np.abs(clean[ws, ts]).max()), 30.0)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    _syml(ax[0], noisy[ws, ts], "noisy (+coherent+intrinsic)", vmax, nblk)
    _syml(ax[1], c1[ws, ts], f"after 1 pass  (stripe {m1['stripe']:.3f}, kept {k1:,})", vmax, nblk)
    _syml(ax[2], c2[ws, ts], f"after 2 passes (stripe {m2['stripe']:.3f}, kept {k2:,})", vmax, nblk)
    im = _syml(ax[3], (c1 - c2)[ws, ts], "1p - 2p  (coherent the 2nd pass removed)", max(vmax * 0.15, 6), nblk)
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    fig.suptitle(f"{pl} plane, event {ev} — smart gate k=3: one vs two passes  "
                 f"(coh_left {m1['coh_left']:.3f}->{m2['coh_left']:.3f}, "
                 f"kept {k1:,}->{k2:,}, F0 loss {m1['signal_lost']*100:.2f}->{m2['signal_lost']*100:.2f}%)",
                 fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, f"multipass_{pl}_ev{ev}.png")
    fig.savefig(out, dpi=115); plt.close(fig); print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, nargs="+", default=[31])
    args = ap.parse_args()
    reg = load_plane_registry("cubic_wireplane_geometry.json")
    cfg = config_from_file(MP.SHARD)
    npz = np.load(MP.NPZ, allow_pickle=True)
    for ev in args.events:
        for pl in ("U", "V", "Y"):
            panel(ev, pl, cfg, reg, npz)


if __name__ == "__main__":
    main()
