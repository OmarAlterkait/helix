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


GS = 64   # gate block width; the artifacts are exactly this wide


def residual_crop(clean, diff, hw=90, ht=220):
    """Window centred on the worst OFF-SIGNAL residual.

    best_crop finds signal, which is the wrong place to look at coherent
    leftovers: in a signal-dense block the common mode is large and the gate
    refuses it at any kgate, so 3.0 and 4.0 look identical there. The leftover
    strips live in the QUIET regions, so centre on the largest |diff| where the
    clean target is zero.
    """
    off = np.where(np.abs(clean) > 0, 0.0, np.abs(diff))
    nw, T = off.shape
    w0, t0 = np.unravel_index(int(np.argmax(off)), off.shape)
    wi = max(0, min(int(w0) - hw // 2, nw - hw))
    ti = max(0, min(int(t0) - ht // 2, T - ht))
    return slice(wi, wi + hw), slice(ti, ti + ht)


def best_crop(clean, hw=90, ht=220):
    """Signal-rich window. Verbatim from research/wire_denoise/viz_2x2.py."""
    en = np.abs(clean)
    nw, T = en.shape
    wc = en.sum(1)
    tc = en.sum(0)
    wi = max(0, min(int(np.argmax(np.convolve(wc, np.ones(hw), 'same'))) - hw // 2, nw - hw))
    ti = max(0, min(int(np.argmax(np.convolve(tc, np.ones(ht), 'same'))) - ht // 2, T - ht))
    return slice(wi, wi + hw), slice(ti, ti + ht)


def symlog_im_crop(ax, img, title, ws, ts, vmax, cbar=True):
    """Cropped panel with 64-wire group boundaries, as research/wire_denoise does.

    The dashed lines matter here: leftover coherent sits in whole `group_size`
    blocks, so the gridlines show at a glance whether a residual strip is a gate
    block or something else.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    norm = SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax, base=10)
    extent = [ws.start, ws.stop, ts.start, ts.stop]
    im = ax.imshow(img.T, aspect="auto", origin="lower", cmap="RdBu_r", norm=norm,
                   extent=extent)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("wire")
    ax.set_ylabel("tick")
    first = ((ws.start // GS) + 1) * GS
    for g in range(first, ws.stop, GS):
        ax.axvline(g, color="gray", lw=0.5, ls="--", alpha=0.5)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="ADC")
    return im


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


def regenerate_planes(shard_path, run, ev, geom_path, npz):
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
    clean = {int(g): x[0].detach().cpu().numpy().copy() for g, x in grids.items()}
    grids = _dops.add_intrinsic_noise(grids, reg, seeds=[seed], incoherent=True,
                                      coherent=True, series_spectrum=spec,
                                      group_size=cfg.group_size)
    peds = {int(g): cfg.pedestals.get(gmap[int(g)].split("_")[-1], 0) for g in grids}
    grids = _dops.digitize(grids, peds, n_bits=12)
    noisy = {int(g): x[0].detach().cpu().numpy() for g, x in grids.items()}
    return noisy, clean, seed


def regenerate_noisy(shard_path, run, ev, geom_path, npz):
    """Back-compat shim: ``(noisy, seed)``."""
    noisy, _clean, seed = regenerate_planes(shard_path, run, ev, geom_path, npz)
    return noisy, seed


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
    ap.add_argument("--clean-source", choices=("corpus", "true"), default="corpus",
                    help="'corpus' = the stored coeff_clean, i.e. the CO-SUPPORTED "
                         "target the FM regresses onto (clean restricted to the "
                         "coordinates removal+threshold kept). 'true' = the "
                         "noise-free image itself, regenerated. They differ: the "
                         "co-supported target is missing ~2.7% of the true signal "
                         "charge, so a diff against it forgives whatever the "
                         "denoiser discarded, and its inverse-DWT leakage spreads "
                         "nonzero pixels far wider (1.85M vs 392k) which distorts "
                         "any mask built from it.")
    ap.add_argument("--no-noisy", action="store_true",
                    help="skip the regenerated panel; corpus-only (3 of 4 panels)")
    ap.add_argument("--crop", action="store_true",
                    help="zoom to the signal-rich window (best_crop) instead of "
                         "the full plane, with 64-wire group boundaries drawn")
    ap.add_argument("--crop-on", choices=("signal", "residual"), default="signal",
                    help="'signal' = best_crop (signal-rich, research convention); "
                         "'residual' = centre on the worst off-signal leftover, "
                         "which is where kgate actually changes the picture")
    ap.add_argument("--crop-at", type=int, nargs=2, metavar=("WIRE", "TICK"),
                    default=None,
                    help="force the crop's start indices. Needed to compare two "
                         "corpora: each would otherwise pick its own worst-residual "
                         "window and the panels would not be the same patch.")
    ap.add_argument("--hw", type=int, default=90, help="crop width in wires")
    ap.add_argument("--ht", type=int, default=220, help="crop height in ticks")
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

    import json as _json
    import h5py as _h5
    with _h5.File(noisy_shard, "r") as _f:
        kg_lbl = _json.loads(_f["config"].attrs["removal_json"]).get("kgate", "?")

    removed_all = ce.reconstruct_images()
    clean_all = cc.reconstruct_images()

    noisy_all = true_clean_all = None
    if a.clean_source == "true" and a.no_noisy:
        raise SystemExit("--clean-source true needs the regeneration that "
                         "--no-noisy skips")
    if not a.no_noisy:
        # The corpus event carries the SHARD it came from; the noisy image is
        # regenerated from that sensor shard, not from a run-wide index.
        sensor_shard = os.path.join(a.source_root, a.split, src_file)
        noisy_all, true_clean_all, seed = regenerate_planes(
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

    CLEAN_TITLE = ("clean (TRUE noise-free image)" if a.clean_source == "true"
                   else "clean (co-supported target)")

    for gid in a.planes:
        if gid not in clean_all:
            print(f"  plane {gid} not in this shard's plane set {sorted(clean_all)}")
            continue
        cl = (true_clean_all[gid] if a.clean_source == "true" else clean_all[gid])
        rc = removed_all[gid]
        diff = rc - cl
        sig = np.abs(cl) > 0
        # Same F0 the reference figure reports: fraction of clean charge left
        # intact where the (co-supported) clean target is nonzero.
        f0 = 1.0 - np.abs(diff)[sig].sum() / max(np.abs(cl)[sig].sum(), 1e-9)
        vmax = max(float(np.percentile(np.abs(cl[sig]), 99.5)) if sig.any() else 20.0, 30.0)

        if a.crop:
            if a.crop_at:
                ws = slice(a.crop_at[0], a.crop_at[0] + a.hw)
                ts = slice(a.crop_at[1], a.crop_at[1] + a.ht)
            elif a.crop_on == "residual":
                ws, ts = residual_crop(cl, diff, a.hw, a.ht)
            else:
                ws, ts = best_crop(cl, a.hw, a.ht)
            print(f"  crop w[{ws.start}:{ws.stop}] t[{ts.start}:{ts.stop}]")
            cl_v, rc_v = cl[ws, ts], rc[ws, ts]
            no_v = noisy_all[gid][ws, ts] if noisy_all is not None else None
            df_v = rc_v - cl_v
            # vmax from the CROP, not the full plane: a window-local scale is what
            # makes the residual visible at this zoom.
            sg = np.abs(cl_v) > 0
            vmax = max(float(np.percentile(np.abs(cl_v[sg]), 99.5)) if sg.any() else 20.0, 30.0)
            fig, ax = plt.subplots(2, 2, figsize=(11, 8))
            symlog_im_crop(ax[0, 0], cl_v, CLEAN_TITLE, ws, ts, vmax)
            if no_v is not None:
                symlog_im_crop(ax[0, 1], no_v, "noisy: +coherent +intrinsic (seed-matched)",
                               ws, ts, vmax)
            else:
                ax[0, 1].set_axis_off()
                ax[0, 1].text(0.5, 0.5, "noisy not stored in corpus", ha="center",
                              va="center", fontsize=11)
            symlog_im_crop(ax[1, 0], rc_v, "removed: corpus coeff -> DWT recon", ws, ts, vmax)
            symlog_im_crop(ax[1, 1], df_v, "diff (recon - clean)", ws, ts,
                           max(vmax * 0.5, 20.0))
            fig.suptitle(f"{src_file} evt{ev:03d}  plane_gid {gid}  ZOOM "
                         f"w[{ws.start}:{ws.stop}] t[{ts.start}:{ts.stop}]  "
                         f"kgate={kg_lbl}  clean={a.clean_source}  on={a.crop_on}  F0={f0:.3f} (full plane)",
                         fontweight="bold")
            tag = f"zoom{a.crop_on[0]}"
        else:
            fig, ax = plt.subplots(2, 2, figsize=(13, 9))
            symlog_im(ax[0, 0], cl, CLEAN_TITLE, vmax)
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
                         f"({cl.shape[0]}w x {cl.shape[1]}t)  kgate={kg_lbl}  F0={f0:.3f}",
                         fontweight="bold")
            tag = "full"
        fig.tight_layout()
        out = f"{a.outdir}/{tag}_{a.clean_source}_ev{ev:03d}_gid{gid}.png"
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
