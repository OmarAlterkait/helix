"""Masked-reconstruction event displays — the FM's actual pretraining task.

The task is masked DENOISING, not self-reconstruction. ``CoeffTokenize`` is
configured ``clean_part='coeff_clean'``, so the model sees the *denoised*
coefficients (``inp``) with a fraction of tokens hidden, and is scored against
the *clean* coefficients (``tgt``) at the same coordinates. A display that shows
``inp`` as "truth" is showing the input, not the target.

Two mask modes, both drawn by the model's own ``make_mask`` so the picture is the
objective rather than a re-implementation of it:

  ``random``  a fraction (default 0.75, m113's ``--mask``) of tokens anywhere
  ``plane``   whole plane(s) hidden — the cross-plane task that ``plane_frac=0.1``
              mixes in, where the only route back is the other views

Panels follow ``scripts/viz_2x2_corpus.py`` — wire x tick images in ADC after an
inverse DWT, ``SymLogNorm(linthresh=2)/RdBu_r``, 64-wire gate-block gridlines —
so these figures sit next to the denoising 2x2s without a change of units. The
coefficient domain is where the loss lives, but it is not where anyone can see
whether a track came back.

All four panels are reconstructed from the TOKEN support (bands 0..n_bands-1),
so D1 — which the tokenizer drops — is absent from every panel equally. That is
a property of what the model can see, not of the corpus.

Usage::

    python3 scripts/viz_mask_recon.py \\
        --checkpoint /sdf/data/neutrino/omara/exp/export/coeff-fm-train-r1-ema \\
        --corpus /sdf/data/neutrino/omara/coeff_tpc_r1/run_0027575715 \\
        --truth  /sdf/data/neutrino/omara/coeff_tpc/run_0027575715/truth/probe_truth_probe.h5 \\
        --tag r1-ema --events 0 1 2 --mode both --out FIGS
"""

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The panel helpers live next door and are shared deliberately: these figures
# must be readable beside the denoising 2x2s, which means the SAME linthresh,
# colormap, crop and gate-block gridlines, not a lookalike set.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz_2x2_corpus import best_crop, residual_crop, symlog_im_crop, symlog_im


def _position_of(shard, event_id):
    """POSITION of ``event_id`` in ``shard`` — not the id itself.

    ``read_coeff_event`` slices by position, and the two diverge on any shard
    whose source events are not contiguous. Same guard as
    ``scripts/run_probe.py``; see the note there for the shard that exposed it.
    """
    import h5py
    with h5py.File(shard, "r") as f:
        ids = f["ident"]["event"][:]
    pos = int(np.searchsorted(ids, event_id))
    if pos >= len(ids) or int(ids[pos]) != int(event_id):
        raise SystemExit(
            f"{shard}: no event with id {event_id} (shard holds {len(ids)} "
            f"events, {ids.min()}..{ids.max()})")
    return pos


def _load_identity(truth_path):
    """The probe holdout's (run, source_file, event) list, in artifact order."""
    import h5py
    with h5py.File(truth_path, "r") as f:
        g = f["ident"]
        return [(r.decode() if isinstance(r, bytes) else str(r),
                 s.decode() if isinstance(s, bytes) else str(s), int(e))
                for r, s, e in zip(g["run"][:], g["source_file"][:], g["event"][:])]


def _centroids(model):
    """Per-band E[raw/sigma | bin] for the categorical head.

    Just the buffer: `set_bins` derives `bin_cent_ratio` when a sidecar does not
    measure it, so it is finite whenever `bin_edges` is, and a pre-fix export
    gets it derived on load by `checkpoint._backfill_centroids`. This used to
    carry its own fallback to the closed form, which made three copies of "how
    do I get a centroid" across the tree — and the copy the evaluator used was
    the one that silently returned nothing.

    NEVER build asinh centres and sinh them: the categorical read-back is
    sum_k p_k * centroid_k, and applying sinh to that mean is Jensen-biased ~31%
    low. Reference: fm/train.py:125, fm/viz_cat.py:25.
    """
    import torch
    cr = model.bin_cent_ratio.detach().float().cpu()
    assert torch.isfinite(cr).all(), (
        "bin_cent_ratio is not finite — set_bins() was bypassed or the sidecar "
        "carried a NaN table; refusing to decode charge through it")
    return cr.numpy()


def _images(ce, tok, occ_mask, values, gids, norm_sigma, pcfg, space="asinh"):
    """(cell, slot) grid -> {gid: (n_wires, n_time) ADC image} via the real inverse.

    ``_rows_from_grid`` is the tokenizer's own inverse (the one ``detokenize``
    and ``decode_prediction`` are both built on), so the coefficients that come
    back are the ones that went in — no second derivation to drift.
    """
    from helix.model.tokenize import _rows_from_grid
    rows = _rows_from_grid(occ_mask, values, tok["cell_band"], tok["cell_gid"],
                           tok["cell_wb"], tok["cell_tb"],
                           gids=gids, norm_sigma=norm_sigma, cfg=pcfg, space=space)
    ev = replace(ce, band=rows["band"], plane_gid=rows["plane_gid"],
                 wire=rows["wire"], tau=rows["tau"], value=rows["value"])
    return ev.reconstruct_images(), int(rows["value"].shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="pimm export dir")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--dataset-name", default="sim_wire")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--events", type=int, nargs="+", default=[0])
    ap.add_argument("--mode", default="both",
                    choices=["random", "plane", "plane_any", "both"])
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--n-planes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--planes", type=int, nargs="*", default=None,
                    help="planes to DRAW; -1 = every plane in the event")
    ap.add_argument("--mask-plane", nargs="*", default=None,
                    help="mode=plane only: HIDE exactly these plane_gids, one case "
                         "each (or 'all': every plane in the event) instead of "
                         "letting make_mask draw one. --planes chooses what is "
                         "DRAWN, which is a different thing and does not control "
                         "the mask. Sweeping --seed varies the hidden plane only "
                         "at random -- six draws over six planes cover about four "
                         "distinct ones -- so this is what gives every plane a turn.")
    ap.add_argument("--crop-on", default="signal", choices=["signal", "residual"])
    ap.add_argument("--zoom", action="store_true",
                    help="crop to a window instead of the whole plane; the "
                         "full plane is the default because that is what the "
                         "corpus 2x2 figures show. Same as --views zoom.")
    ap.add_argument("--views", nargs="+", default=None, choices=["full", "zoom"],
                    help="draw each case in these views (e.g. full zoom)")
    ap.add_argument("--value", default="mean", choices=["mode", "mean"],
                    help="categorical read-out. 'mean' closes charge and is "
                         "what the reference reports; 'mode' is the reference's "
                         "bright-pixel-fidelity read-out.")
    ap.add_argument("--wire", action="store_true",
                    help="also draw, for the wire with the most hidden charge, "
                         "the truth waveform against the predicted mean and its "
                         "exact +-1/2 sigma band, plus the predicted distribution "
                         "of a few of its masked coefficients")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-t", default=None, choices=("grid_center", "centroid"),
                    help="tokenizer cell_t, REQUIRED when the checkpoint records no "
                         "tokenizer. helix configs train grid_center.")
    ap.add_argument("--per-event-dirs", action="store_true",
                    help="write each event's figures under <out>/ev<i>/")
    a = ap.parse_args()
    views = a.views or (["zoom"] if a.zoom else ["full"])

    import torch
    import h5py
    from helix.core.coeff_io import read_coeff_event
    from helix.model.tokenize import (PatchConfig, assemble, to_fm,
                                      decode_categorical)
    from helix.model.checkpoint import patch_config_from_checkpoint
    from helix.probe.features import load_probe_model

    os.makedirs(a.out, exist_ok=True)
    corpus = os.path.abspath(a.corpus)
    ident = _load_identity(a.truth)
    pcfg = patch_config_from_checkpoint(a.checkpoint, cell_t=a.cell_t)
    src = "checkpoint" if pcfg is not None else "--cell-t"
    if pcfg is None:
        if getattr(a, "cell_t", None) is None:
            raise SystemExit(
                f"{a.checkpoint} records no tokenizer, and --cell-t was not given.\n"
                "Refusing to guess. The old fallback was `... or PatchConfig()`, which\n"
                "silently chose cell_t='centroid' while every helix config trains\n"
                "'grid_center' -- they differ on 94.06% of cells (mean |delta| 19.5\n"
                "ticks), so the model would be scored on a time coordinate it never saw.\n"
                "Pass --cell-t grid_center, or read `cell_t` out of the run's config.py.")
        pcfg = PatchConfig(cell_t=a.cell_t)
    print(f"tokenizer: cell_t={pcfg.cell_t} pw={pcfg.pw} pt={pcfg.pt} (from {src})", flush=True)
    print(f"tokenizer: cell_t={pcfg.cell_t} pw={pcfg.pw} pt={pcfg.pt} "
          f"n_bands={pcfg.n_bands}", flush=True)

    model, meta = load_probe_model(a.checkpoint, weights=a.weights)
    dev = next(model.parameters()).device
    print(f"model {os.path.basename(a.checkpoint)} on {dev}", flush=True)
    centroids = _centroids(model)
    modes = ["random", "plane"] if a.mode == "both" else [a.mode]

    for ev_i in a.events:
        run, src, ev = ident[ev_i]
        shard_tag = src.replace("sim_wire_sensor_", "").replace(".h5", "")
        shard = os.path.join(corpus, f"{a.dataset_name}_coeff_{shard_tag}.h5")
        clean_shard = os.path.join(corpus,
                                   f"{a.dataset_name}_coeff_clean_{shard_tag}.h5")
        with h5py.File(shard, "r") as f:
            c = f["config"]
            gids, nw = c["gids"][:], c["n_wires"][:]
            bl, ns = c["band_lengths"][:], c["norm_sigma"][:]
        pos = _position_of(shard, ev)
        ce = read_coeff_event(shard, pos)                       # denoised (input)
        # The clean shard is values-only and co-supported: same coordinates, so
        # its digest must match or every target row is misaligned. read_coeff_event
        # enforces that; passing `ce` is what supplies the coordinates.
        cc = read_coeff_event(clean_shard, pos, coords_from=ce)  # clean (target)

        tok = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value,
                       gids=gids, n_wires=nw, band_lengths=bl, norm_sigma=ns,
                       cfg=pcfg, value_clean=cc.value)
        B = to_fm(tok)
        Bt = {k: (torch.as_tensor(v).to(dev) if isinstance(v, np.ndarray) else v)
              for k, v in B.items()}
        Bt.setdefault("n_cells", Bt["plane_id"].shape[0])
        print(f"\nevent {ev_i} (sim {ev}, shard {shard_tag}): "
              f"{tok['cell_band'].shape[0]:,} cells, {ce.n_coeff:,} coeffs",
              flush=True)

        occ_t = tok["occ"].astype(bool)
        valid = tok["valid"].astype(bool)
        inp, tgt = tok["inp"], tok["tgt"]
        ev_out = os.path.join(a.out, f"ev{ev_i}") if a.per_event_dirs else a.out
        os.makedirs(ev_out, exist_ok=True)

        # One case per (mode, hidden plane). --mask-plane turns plane mode into
        # one deterministic case per named plane instead of make_mask's draw.
        present = sorted(np.unique(tok["cell_gid"]).tolist())
        cases = []
        for mode in modes:
            if mode == "plane" and a.mask_plane:
                hide = present if a.mask_plane == ["all"] else [int(g) for g in a.mask_plane]
                cases += [("plane", g) for g in hide]
            else:
                cases.append((mode, None))

        clean_cache = None
        for mode, hide in cases:
            gen = torch.Generator(device=dev).manual_seed(a.seed)
            m = model.make_mask(Bt, mode=mode, ratio=a.ratio,
                                n_planes=a.n_planes, gen=gen)
            if hide is not None:
                # Deterministic override of make_mask's randperm draw. Same shape
                # and dtype, so everything downstream is unchanged; only WHICH
                # plane is hidden differs.
                m = Bt["plane_id"] == int(hide)
                if not bool(m.any()):
                    print(f"    plane {hide}: absent from this event, "
                          f"skipping", flush=True)
                    continue
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=dev.type == "cuda"):
                occ_logit, logits, _ = model.raw_heads(Bt, m)
            occ_logit = occ_logit.float().cpu().numpy()
            # UNITS OF SIGMA (space="ratio"), never asinh — see _centroids.
            logits = logits.float().cpu().numpy()
            pred = decode_categorical(logits, tok["cell_band"], centroids,
                                      readout=a.value)
            if not a.wire:
                del logits
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            mask = m.cpu().numpy().astype(bool)[:, None]        # (n_cells, 1)
            emit = (occ_logit > 0.0) & valid                    # decode_prediction's rule

            # The three coefficient sets, all on the token support:
            #   clean   the target, everywhere it exists
            #   seen    the input, visible cells only  (what the model is given)
            #   recon   input on visible cells, model output on masked ones
            occ_clean = occ_t
            occ_seen = occ_t & ~mask
            occ_recon = (occ_t & ~mask) | (emit & mask)
            # ONE space. `pred` is in units of sigma (the categorical read-out);
            # `inp` is the asinh token. sinh() the visible side rather than
            # asinh-ing the prediction — the latter would re-introduce exactly
            # the nonlinearity this decode path exists to avoid.
            v_recon = np.where(mask, pred, np.sinh(inp))

            if clean_cache is None:                     # same for every case
                clean_cache = _images(ce, tok, occ_clean, tgt, gids, ns, pcfg)
            img_clean, n_clean = clean_cache
            img_seen, n_seen = _images(ce, tok, occ_seen, inp, gids, ns, pcfg)
            img_recon, n_recon = _images(ce, tok, occ_recon, v_recon, gids, ns,
                                         pcfg, space="ratio")

            # ---- numbers that belong with the picture ---------------------
            sv = mask & valid & occ_t
            # in sigma units on BOTH sides: pred is a ratio, tgt is an asinh token
            rmse = (float(np.sqrt(np.mean((pred[sv] - np.sinh(tgt[sv])) ** 2)))
                    if sv.any() else np.nan)
            so = mask & valid
            tp = int((emit & occ_t & so).sum())
            fp = int((emit & ~occ_t & so).sum())
            fn = int((~emit & occ_t & so).sum())
            prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            print(f"  [{mode}] masked {mask.mean():.3f} of cells | coeffs "
                  f"clean {n_clean:,} seen {n_seen:,} recon {n_recon:,} | "
                  f"RMSE {rmse:.4f} sigma | occ P {prec:.3f} R {rec:.3f}", flush=True)

            if hide is not None:
                planes = [int(hide)]
            elif a.planes is not None:
                planes = present if a.planes == [-1] else a.planes
            elif mode == "plane":
                planes = sorted(np.unique(tok["cell_gid"][mask[:, 0]]).tolist())
            else:
                gu, cnt = np.unique(tok["cell_gid"], return_counts=True)
                planes = [int(gu[np.argmax(cnt)])]
            print(f"  [{mode}] planes {planes}", flush=True)

            for gid in planes:
                if gid not in img_clean:
                    print(f"    plane {gid}: not in this event, skipped", flush=True)
                    continue
                for view in views:
                    av = argparse.Namespace(**{**vars(a), "zoom": view == "zoom",
                                               "out": ev_out})
                    _draw(av, gid, img_clean[gid], img_seen[gid], img_recon[gid],
                          ev_i, ev, mode, rmse, prec, rec, float(mask.mean()))
                if a.wire:
                    _wire_figure(argparse.Namespace(**{**vars(a), "out": ev_out}),
                                 gid, ev_i, ev, mode, ce, tok, gids, ns, pcfg,
                                 logits, occ_logit, centroids, mask, pred,
                                 img_clean[gid], img_seen[gid], img_recon[gid], bl)
            if a.wire:
                del logits


_PSI = {}


def _synthesis_matrix(basis, band_lengths, n_bands):
    """Psi: one row per kept coefficient position (band, tau), one column per
    tick -- the waveform one unit coefficient produces. The inverse DWT is
    linear, so a wire's waveform is c @ Psi. Built once by inverse-transforming
    unit impulses with the SAME transform reconstruct() uses; it depends only on
    the basis and the readout length, never on the event."""
    import pywt
    key = (basis.wavelet, basis.mode, basis.n_ticks_raw, tuple(int(b) for b in band_lengths), n_bands)
    if key not in _PSI:
        bl = [int(b) for b in band_lengths]
        rows, offs = [], []
        for b in range(n_bands):
            n = bl[b]
            coeffs = [np.zeros((n, bl[j])) for j in range(len(bl))]
            coeffs[b][np.arange(n), np.arange(n)] = 1.0
            rows.append(pywt.waverec(coeffs, basis.wavelet, mode=basis.mode,
                                     axis=-1)[:, :basis.n_ticks_raw])
            offs.append(sum(bl[:b]))
        _PSI[key] = (np.concatenate(rows, 0), offs)
    return _PSI[key]


def _wire_figure(a, gid, ev_i, ev, mode, ce, tok, gids, ns, pcfg, logits, occ_logit,
                 centroids, mask, pred, clean, seen, recon, band_lengths):
    """One wire, with the model's own uncertainty carried into ADC exactly.

    The head gives every masked coefficient a full distribution: occupancy
    probability q and a categorical over bins with centroids c. So its mean is
    q*sum(p c) and its variance q*sum(p c^2) - mean^2 -- no sampling. The wire's
    waveform is linear in its coefficients, x = c @ Psi, hence

        E[x]   = mu  @ Psi          Var[x] = v @ Psi**2

    exactly, under the one assumption the head itself makes: coefficients are
    predicted independently. Visible coefficients are known (variance 0).

    Second figure: the predicted distribution of a few masked coefficients on
    this wire (the wavelet runs along time, so each belongs to one wire).
    """
    from helix.model.tokenize import sigma_for_rows
    pw, pt, nb = pcfg.pw, pcfg.pt, pcfg.n_bands
    w = int(np.abs(clean - seen).sum(1).argmax())       # most hidden charge
    Psi, offs = _synthesis_matrix(ce.basis, band_lengths, nb)
    mu = np.zeros(Psi.shape[0])
    var = np.zeros(Psi.shape[0])

    # every token slot on this wire: cells of this plane in wire block w // pw,
    # slot row w % pw
    cells = np.where((tok["cell_gid"] == gid) & (tok["cell_wb"] == w // pw))[0]
    sl = np.arange((w % pw) * pt, (w % pw) * pt + pt)
    picks = []
    for ci in cells:
        b = int(tok["cell_band"][ci])
        sig = float(sigma_for_rows(np.array([gid]), np.array([b]), gids, ns)[0])
        for s_ in sl:
            if not tok["valid"][ci, s_]:
                continue
            k = offs[b] + int(tok["cell_tb"][ci]) * pt + s_ % pt
            if not mask[ci, 0]:                         # visible: known exactly
                if tok["occ"][ci, s_]:
                    mu[k] = np.sinh(tok["inp"][ci, s_]) * sig
                continue
            x = logits[ci, s_].astype(np.float64)
            p = np.exp(x - x.max()); p /= p.sum()
            c = np.asarray(centroids, np.float64)[b]
            q = 1.0 / (1.0 + np.exp(-float(occ_logit[ci, s_])))
            m1, m2 = q * (p @ c), q * (p @ c ** 2)
            mu[k], var[k] = m1 * sig, (m2 - m1 ** 2) * sig ** 2
            if tok["occ"][ci, s_]:
                picks.append((abs(float(tok["tgt"][ci, s_])), ci, s_, b, p))
    mean = mu @ Psi
    std = np.sqrt(np.maximum(var @ Psi ** 2, 0.0))

    tru = clean[w]
    thr = 0.05 * max(np.abs(tru).max(), np.abs(mean).max(), 1e-9)
    on = np.abs(tru) > thr
    z = np.abs(tru - mean) / np.maximum(std, 1e-9)
    cov1 = float((z <= 1)[on].mean()) if on.any() else np.nan
    cov2 = float((z <= 2)[on].mean()) if on.any() else np.nan
    nz = np.where((np.abs(tru) > thr) | (np.abs(mean) > thr))[0]
    t0, t1 = ((max(nz.min() - 40, 0), min(nz.max() + 40, tru.size)) if nz.size
              else (0, tru.size))
    t = np.arange(t0, t1)

    sub = (f"random {a.ratio:.0%} of tokens hidden" if mode == "random"
           else f"{a.n_planes} whole plane(s) hidden")
    stem = os.path.join(a.out, f"wire_{a.tag}_ev{ev_i}_{mode}_plane{gid}_w{w}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(t, mean[t0:t1] - 2 * std[t0:t1], mean[t0:t1] + 2 * std[t0:t1],
                    color="tab:red", alpha=0.15, lw=0, label="predicted ±2σ")
    ax.fill_between(t, mean[t0:t1] - std[t0:t1], mean[t0:t1] + std[t0:t1],
                    color="tab:red", alpha=0.3, lw=0, label="predicted ±1σ")
    ax.plot(t, seen[w, t0:t1], color="0.6", lw=0.8, ls=":", label="model input (visible only)")
    ax.plot(t, tru[t0:t1], color="k", lw=1.3, label="clean truth")
    ax.plot(t, mean[t0:t1], color="tab:red", lw=1.1, label="predicted mean")
    ax.set_xlabel("time tick"); ax.set_ylabel("ADC")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"{a.tag}  evt{ev:03d}  {sub}\nplane_gid {gid}, wire {w}: truth within "
                 f"±1σ on {cov1:.0%} and ±2σ on {cov2:.0%} of signal ticks "
                 f"(68% / 95% if calibrated)\nexact mean and variance through the "
                 f"linear inverse DWT; coefficients independent, as the head predicts them",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{stem}_waveform.png", dpi=130)
    plt.close(fig)

    picks.sort(key=lambda r: -r[0])
    if len(picks) > 4:
        picks = picks[:3] + [picks[len(picks) // 2]]
    if picks:
        fig, axs = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.2),
                                squeeze=False)
        for j, (_, ci, s_, b, p) in enumerate(picks):
            axj = axs[0, j]
            tau = int(tok["cell_tb"][ci]) * pt + s_ % pt
            xs = np.arcsinh(np.asarray(centroids, np.float64)[b])
            axj.step(xs, p, where="mid", color="tab:blue", lw=1.0,
                     label="predicted distribution")
            axj.axvline(float(tok["tgt"][ci, s_]), color="k", lw=1.3, label="truth")
            axj.axvline(float(np.arcsinh(pred[ci, s_])), color="tab:red", lw=1.0,
                        ls="--", label="predicted mean")
            lo, hi = np.searchsorted(np.cumsum(p), [0.001, 0.999])
            axj.set_xlim(xs[max(lo - 3, 0)], xs[min(hi + 3, xs.size - 1)])
            axj.set_xlabel("asinh(value / sigma)")
            axj.set_title(f"band {b}, tick {tau}", fontsize=9)
            if j == 0:
                axj.set_ylabel("probability")
                axj.legend(fontsize=8)
        fig.suptitle(f"{a.tag}  evt{ev:03d}  {sub}  plane_gid {gid}, wire {w}: "
                     f"predicted distribution of masked coefficients on this wire "
                     f"(brightest first)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(f"{stem}_coefficients.png", dpi=130)
        plt.close(fig)
    print(f"    wire {w}: within 1σ {cov1:.2f}, 2σ {cov2:.2f}  wrote "
          f"{os.path.basename(stem)}_{{waveform,coefficients}}.png", flush=True)


def _f0(clean, other):
    """Fraction of clean charge left intact where the clean target is nonzero.

    The same F0 ``viz_2x2_corpus`` reports for the denoising panels, computed the
    same way — so a masked-reconstruction figure and a denoising figure carry the
    same number and can be read against each other.
    """
    sig = np.abs(clean) > 0
    if not sig.any():
        return float("nan")
    return 1.0 - np.abs(other - clean)[sig].sum() / max(np.abs(clean)[sig].sum(), 1e-9)


def _draw(a, gid, clean, seen, recon, ev_i, ev, mode, rmse, prec, rec, frac):
    diff = recon - clean
    sig = np.abs(clean) > 0
    # Scale convention taken from viz_2x2_corpus: 99.5th percentile of the clean
    # target WHERE IT IS NONZERO, floored at 30 ADC, and the difference panel at
    # half that. Percentile over the whole plane instead would be dominated by
    # the empty background and collapse the scale.
    vmax = max(float(np.percentile(np.abs(clean[sig]), 99.5)) if sig.any() else 20.0, 30.0)
    vdiff = max(vmax * 0.5, 20.0)
    f0 = _f0(clean, recon)
    sub = (f"random {a.ratio:.0%} of tokens hidden" if mode == "random"
           else f"{a.n_planes} whole plane(s) hidden — cross-plane task")
    titles = ["clean (truth, co-supported target)",
              f"model input — {sub}",
              "reconstruction (model on masked, input on visible)",
              "diff (recon - clean)"]
    head = (f"{a.tag}  evt{ev:03d}  plane_gid {gid}  MASKED RECONSTRUCTION — {sub}\n"
            f"masked {frac:.1%} of cells  |  F0={f0:.3f}  |  token RMSE "
            f"{rmse:.3f}  |  occupancy P {prec:.2f} R {rec:.2f}")

    if not a.zoom:
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        for axi, img, t, vm in zip(ax.ravel(), (clean, seen, recon, diff),
                                   titles, (vmax, vmax, vmax, vdiff)):
            symlog_im(axi, img, t, vm)
        fig.suptitle(head, fontweight="bold", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = os.path.join(a.out, f"maskrecon_{a.tag}_ev{ev_i}_{mode}_plane{gid}.png")
    else:
        ws, ts = (best_crop(clean) if a.crop_on == "signal"
                  else residual_crop(clean, diff))
        cl_v, sn_v, rc_v = clean[ws, ts], seen[ws, ts], recon[ws, ts]
        df_v = rc_v - cl_v
        sg = np.abs(cl_v) > 0
        # Window-local scale, as the corpus zoom does: a full-plane vmax makes
        # the residual invisible at this magnification.
        vmax = max(float(np.percentile(np.abs(cl_v[sg]), 99.5)) if sg.any() else 20.0, 30.0)
        vdiff = max(vmax * 0.5, 20.0)
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        for axi, img, t, vm in zip(ax.ravel(), (cl_v, sn_v, rc_v, df_v),
                                   titles, (vmax, vmax, vmax, vdiff)):
            symlog_im_crop(axi, img, t, ws, ts, vm)
        fig.suptitle(head + f"  ZOOM w[{ws.start}:{ws.stop}] t[{ts.start}:{ts.stop}]",
                     fontweight="bold", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = os.path.join(a.out,
                         f"maskrecon_{a.tag}_ev{ev_i}_{mode}_plane{gid}_zoom{a.crop_on[0]}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print(f"    F0={f0:.4f}  wrote {os.path.basename(p)}", flush=True)


if __name__ == "__main__":
    main()
