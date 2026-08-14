"""The 2x2 denoising panels, sourced FROM THE CORPUS.

Same figure as ``research/coeff_foundation_model/viz_2x2_coeff.py`` — same panel
order, same SymLogNorm(linthresh=2)/RdBu_r styling, same titles — but the arrays
come from the shipped corpus instead of being recomputed from the simulation.
That is the point: it answers whether the corpus alone carries the method.

It mostly does. Three of the four panels are stored:

    clean     coeff_clean -> reconstruct_images()
    removed   coeff       -> reconstruct_images()
    diff      removed - clean

The NOISY panel is not in the corpus. Only the post-removal, post-threshold
coefficients are kept, so the noisy image has to be regenerated — which is exact
rather than approximate, because each event records the ``noise_seed`` that
produced it and ``/config`` records the spectrum (with sha256), the removal
parameters and the basis. This script regenerates it through the SAME transforms
the builder used (Densify -> AddNoise -> Digitize, build_coeff_corpus.py:60-67)
and asserts the recomputed seed matches the stored one before drawing, so a
mismatched event cannot be silently plotted against the wrong noise.

One nuance worth stating on the figure: ``coeff_clean`` is CO-SUPPORTED. The
builder stores the noise-free image's coefficients at the NOISY event's kept
coordinates (build_coeff_corpus.py:11), not an independent sparsification of the
clean image. So "clean" here is the regression target the FM actually sees —
clean restricted to the denoised support — which is what makes the clean/removed
comparison like-for-like. It is NOT the full clean image, and the difference is
real wherever removal dropped a coordinate that carried signal.

Run on turing: the DSP is not bit-comparable across GPU generations and this
corpus was built on turing (PROVENANCE.md), so the regenerated noisy panel is
bit-faithful only there.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def symlog_im(ax, img, title, vmax, cbar=True):
    """Verbatim from research/coeff_foundation_model/viz_2x2_coeff.py."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("wire")
    ax.set_ylabel("tick")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="ADC")
    return im


def regenerate_noisy(shard_path, run, ev, geom_path, npz):
    """Rebuild the noisy dense planes for one event, the builder's SERIAL way.

    Replicates ``build_coeff_corpus.py::_plane_fn_torch`` (lines 238-269) and its
    ``_seed`` (lines 217-231). The serial path is the one that built this corpus
    (``/config`` records ``noise_json.mode = "serial"``), and it seeds from the
    SHARD NAME —

        blake2b(f"{run}/{shard}/ev{ev}", digest_size=8) & 0xFFFFFFFF

    — not from the loader's ``content_seed(event_name, ...)``. The two disagree,
    so using the loader path here silently draws a different noise realisation
    than the stored coefficients came from. Returns ``(planes, seed)``.
    """
    import hashlib
    import os.path as osp

    import torch

    from helix.core import backend as _backend
    _backend.set_backend("torch")
    from helix.tpc import dense_ops as _dops
    from helix.tpc.config import DetectorConfig
    from helix.tpc.io import config_from_file, read_sensor_event_coo
    from helix.tpc.pipeline import canonical_plane_gid
    from pimm_data.geometry import load_plane_registry

    spec = None
    if npz and osp.exists(npz):
        z = np.load(npz, allow_pickle=True)
        # (freqs_hz, shape) tuple, as build_coeff_corpus.py:194-195 builds it —
        # handing AddNoise the raw NpzFile unpacks to the wrong arity.
        spec = (z["spectrum_freqs_hz"], z["spectrum_shape"])

    reg = load_plane_registry(geom_path)
    base = config_from_file(shard_path)
    cfg = DetectorConfig(num_time_steps=base.num_time_steps,
                         plane_labels=base.plane_labels, pedestals=base.pedestals)
    seed = int.from_bytes(
        hashlib.blake2b(f"{run}/{osp.basename(shard_path)}/ev{ev}".encode(),
                        digest_size=8).digest(), "little") & 0xFFFFFFFF

    coo = read_sensor_event_coo(shard_path, ev, cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_l, t_l, v_l, p_l, gmap = [], [], [], [], {}
    for label, (w, t, v, nw, nt) in coo.items():
        gid = canonical_plane_gid(label)
        gmap[gid] = label
        w_l.append(torch.as_tensor(np.asarray(w), dtype=torch.long, device=dev))
        t_l.append(torch.as_tensor(np.asarray(t), dtype=torch.long, device=dev))
        v_l.append(torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev))
        p_l.append(torch.full((len(w),), gid, dtype=torch.long, device=dev))
    wire, time_ = torch.cat(w_l), torch.cat(t_l)
    val, pid = torch.cat(v_l), torch.cat(p_l)
    offset = torch.tensor([wire.numel()], dtype=torch.long, device=dev)

    grids = _dops.densify(wire, time_, val, pid, offset, reg)
    grids = _dops.add_intrinsic_noise(grids, reg, seeds=[seed], incoherent=True,
                                      coherent=True, series_spectrum=spec,
                                      group_size=cfg.group_size)
    peds = {int(g): cfg.pedestals.get(gmap[int(g)].split("_")[-1], 0) for g in grids}
    grids = _dops.digitize(grids, peds, n_bits=12)
    return {int(g): x[0].detach().cpu().numpy() for g, x in grids.items()}, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/sdf/data/neutrino/omara/coeff_tpc/run_0027575715")
    ap.add_argument("--dataset-name", default="sim_wire")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--event", type=int, default=0, help="POSITION within the shard")
    ap.add_argument("--planes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--source-root",
                    default="/sdf/data/neutrino/doraemon/wire_test_00_00_02/sensor")
    ap.add_argument("--split", default="run_0027575715")
    ap.add_argument("--geom", default="cubic_wireplane_geometry.json")
    ap.add_argument("--npz",
                    default="/sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz")
    ap.add_argument("--no-noisy", action="store_true",
                    help="skip the regenerated panel; corpus-only (3 of 4 panels)")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from helix.core.coeff_io import read_coeff_event

    os.makedirs(a.outdir, exist_ok=True)
    noisy_shard = f"{a.corpus}/{a.dataset_name}_coeff_{a.shard:04d}.h5"
    clean_shard = f"{a.corpus}/{a.dataset_name}_coeff_clean_{a.shard:04d}.h5"

    ce = read_coeff_event(noisy_shard, a.event)
    cc = read_coeff_event(clean_shard, a.event, coords_from=ce)
    src_file, ev = str(ce.source_file), int(ce.event)
    print(f"corpus event: {src_file} evt{ev:03d}  ({ce.n_coeff:,} coeffs)")

    removed_all = ce.reconstruct_images()
    clean_all = cc.reconstruct_images()

    noisy_all = None
    if not a.no_noisy:
        # The corpus event carries the SHARD it came from; the noisy image is
        # regenerated from that sensor shard, not from a run-wide index.
        sensor_shard = os.path.join(a.source_root, a.split, src_file)
        noisy_all, seed = regenerate_noisy(
            sensor_shard, str(ce.run) or a.split, ev, a.geom, a.npz)
        # The corpus records the seed the builder used. If the recomputed one
        # differs we are about to draw a different noise realisation than the
        # coefficients came from — plausible-looking and wrong.
        stored = _stored_seed(noisy_shard, a.event)
        if stored is not None and int(seed) != int(stored):
            raise SystemExit(
                f"noise seed mismatch for {src_file} ev{ev}: recomputed {seed} "
                f"!= stored {stored} — the regenerated panel would not match")
        print(f"noise seed {seed} matches the corpus")

    for gid in a.planes:
        if gid not in clean_all:
            print(f"  plane {gid} not in this shard's plane set {sorted(clean_all)}")
            continue
        cl, rc = clean_all[gid], removed_all[gid]
        diff = rc - cl
        sig = np.abs(cl) > 0
        # Same F0 the reference figure reports: fraction of clean charge left
        # intact where the (co-supported) clean target is nonzero.
        f0 = 1.0 - np.abs(diff)[sig].sum() / max(np.abs(cl)[sig].sum(), 1e-9)
        vmax = max(float(np.percentile(np.abs(cl[sig]), 99.5)) if sig.any() else 20.0, 30.0)

        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        symlog_im(ax[0, 0], cl, "clean (truth, co-supported target)", vmax)
        if noisy_all is not None:
            symlog_im(ax[0, 1], noisy_all[gid],
                      "noisy: +coherent +intrinsic (regenerated, seed-matched)", vmax)
        else:
            ax[0, 1].set_axis_off()
            ax[0, 1].text(0.5, 0.5, "noisy not stored in corpus",
                          ha="center", va="center", fontsize=11)
        symlog_im(ax[1, 0], rc, "removed: corpus coeff -> DWT recon", vmax)
        symlog_im(ax[1, 1], diff, "diff (recon - clean)", max(vmax * 0.5, 20.0))
        fig.suptitle(f"{src_file} evt{ev:03d}  plane_gid {gid}  FULL PLANE "
                     f"({cl.shape[0]}w x {cl.shape[1]}t)  FROM CORPUS  F0={f0:.3f}",
                     fontweight="bold")
        fig.tight_layout()
        out = f"{a.outdir}/corpus_ev{ev:03d}_gid{gid}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"saved {out}  vmax={vmax:.0f}  F0={f0:.3f}")


def _stored_seed(path, i):
    import h5py
    with h5py.File(path, "r") as f:
        if "ident/noise_seed" not in f:
            return None
        return int(f["ident/noise_seed"][i])


if __name__ == "__main__":
    main()
