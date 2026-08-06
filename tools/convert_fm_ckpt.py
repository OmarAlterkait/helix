"""Convert a research FM checkpoint into a self-contained helix checkpoint.

The research trainer (``coeff_foundation_model/fm/mae_ddp.py``) saves a dict of
``{model, ema, step, <arch kwargs...>}`` with DDP-prefixed parameter names, and
for a categorical head it keeps the bin edges in a SEPARATE file
(``tier1_bins.pt``) named only on the command line. Nothing in the checkpoint
records which bins file it was trained with, so a checkpoint that outlives its
sidecar cannot be evaluated at all — the edges are training-set statistics, not
recoverable from the weights.

This tool produces one file that stands on its own::

    {"config": {...},          # everything build_fm() needs
     "state_dict": {...},      # DDP prefix stripped, loads strict=True
     "bins": {...} | None,     # inlined for categorical heads
     "provenance": {...}}      # source path, step, which weights, digest

Architecture is *inferred from the tensors* and then cross-checked against the
checkpoint's own metadata, so a metadata field that disagrees with the weights
is reported rather than silently trusted.

Usage::

    python tools/convert_fm_ckpt.py IN.pt OUT.pt [--ema] [--bins tier1_bins.pt]
    python tools/convert_fm_ckpt.py IN.pt OUT.pt --dry-run     # report only
"""

from __future__ import annotations

import argparse
import hashlib
import os

import torch

# arch keys the research trainer stores alongside the weights
_META_KEYS = ("d", "blocks", "dec_blocks", "heads", "nll", "cond", "dec_mode",
              "mup", "d_base", "ffn_mult", "wire_rope", "n_slot", "n_bins")


def strip_ddp(sd):
    """Drop the ``module.`` prefix DDP adds to every key."""
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def infer_arch(sd, meta=None):
    """Recover build_fm() kwargs from the tensors themselves.

    Returns ``(config, disagreements)``. Shapes are authoritative; ``meta`` (the
    checkpoint's own recorded kwargs) is used only for values that leave no
    trace in the parameter shapes, and any field where the two disagree is
    reported instead of being quietly resolved."""
    meta = dict(meta or {})
    cfg, bad = {}, []

    d = sd["embed.weight"].shape[0]
    n_slot = sd["occ_head.weight"].shape[0]
    cfg.update(d=d, n_slot=n_slot,
               n_band=sd["band_emb.weight"].shape[0],
               n_plane=sd["plane_emb.weight"].shape[0])

    # embed takes [values, occ] concatenated -> 2 * n_slot
    if sd["embed.weight"].shape[1] != 2 * n_slot:
        bad.append(f"embed in_features {sd['embed.weight'].shape[1]} != 2*n_slot "
                   f"{2 * n_slot}")

    # value head width discriminates the three head modes
    vout = sd["val_head.weight"].shape[0]
    if vout == n_slot:
        cfg.update(n_bins=0, nll=False)
    elif vout == 2 * n_slot:
        cfg.update(n_bins=0, nll=True)
    elif vout % n_slot == 0:
        cfg.update(n_bins=vout // n_slot, nll=False)
    else:
        bad.append(f"val_head width {vout} is not a multiple of n_slot {n_slot}")

    cfg["blocks"] = 1 + max((int(k.split(".")[1]) for k in sd if k.startswith("enc.")),
                            default=-1)
    cfg["dec_blocks"] = 1 + max((int(k.split(".")[1]) for k in sd if k.startswith("dec.")),
                                default=-1)
    # a decoder block with its own kv/q projections is a CrossBlock
    cfg["dec_mode"] = "cross" if any(k.startswith("dec.0.kv.") for k in sd) else "self"
    cfg["cond"] = "film" if any(k.startswith("film.") for k in sd) else "none"
    if cfg["cond"] == "film":
        used = [n for n in ("band", "plane", "wire") if f"film.{n}.weight" in sd
                or f"film.{n}.0.weight" in sd]
        cfg["film"] = tuple(used)
        if "film.wire.0.weight" in sd:
            cfg["n_wirefeat"] = sd["film.wire.0.weight"].shape[1]

    # ffn_mult from the first encoder MLP; heads/mup/d_base leave no shape trace
    if "enc.0.mlp.0.weight" in sd:
        cfg["ffn_mult"] = sd["enc.0.mlp.0.weight"].shape[0] // d
    for k in ("heads", "mup", "d_base", "wire_rope", "lam_t", "lam_w"):
        if k in meta:
            cfg[k] = meta[k]
    if "heads" not in cfg:
        bad.append("attention head count is not recoverable from shapes and is "
                   "absent from the checkpoint metadata — pass it explicitly")

    for k, v in cfg.items():                       # cross-check vs recorded meta
        if k in meta and meta[k] != v and k not in ("film",):
            bad.append(f"{k}: tensors say {v!r}, checkpoint metadata says {meta[k]!r}")
    return cfg, bad


def load_bins(path, n_bins):
    bd = torch.load(path, map_location="cpu", weights_only=False)
    edges = bd["edges"]
    if edges.shape[1] != n_bins + 1:
        raise SystemExit(f"bins file {path}: edges {tuple(edges.shape)} inconsistent "
                         f"with n_bins={n_bins} (expected K+1 = {n_bins + 1})")
    return {k: bd[k] for k in ("edges", "cent_asinh", "cent_lin") if k in bd}


def convert(src, dst=None, *, use_ema=False, bins_path=None, verify=True):
    ck = torch.load(src, map_location="cpu", weights_only=False)
    meta = {k: ck[k] for k in _META_KEYS if k in ck}
    if "film" in ck and isinstance(ck["film"], str):
        meta["film"] = tuple(ck["film"].split(","))

    which = "ema" if use_ema else "model"
    if which not in ck:
        raise SystemExit(f"{src}: no '{which}' weights in checkpoint "
                         f"(has: {sorted(k for k in ck if isinstance(ck[k], dict))})")
    sd = strip_ddp(ck[which])

    cfg, bad = infer_arch(sd, meta)
    if bad:
        raise SystemExit("checkpoint is internally inconsistent:\n  - "
                         + "\n  - ".join(bad))

    bins = None
    if cfg.get("n_bins", 0) > 0:
        if bins_path is None:                     # the sidecar the research CLI defaulted to
            guess = os.path.join(os.path.dirname(os.path.abspath(src)), "tier1_bins.pt")
            bins_path = guess if os.path.exists(guess) else None
        if bins_path is None:
            raise SystemExit(
                f"n_bins={cfg['n_bins']} (categorical head) but no bins file found. "
                f"The edges are training-set statistics and are NOT in the "
                f"checkpoint — pass --bins explicitly.")
        bins = load_bins(bins_path, cfg["n_bins"])

    blob = {
        "config": cfg,
        "state_dict": sd,
        "bins": bins,
        "provenance": {
            "source": os.path.abspath(src),
            "weights": which,
            "step": int(ck.get("step", -1)),
            "bins_file": os.path.abspath(bins_path) if bins_path else None,
            "state_digest": _digest(sd),
        },
    }

    if verify:
        from helix.model import build_fm
        model = build_fm(cfg, serial=True)
        model.load_state_dict(sd, strict=True)     # raises on any mismatch
        if bins is not None:
            model.set_bins(bins["edges"], bins.get("cent_asinh"), bins.get("cent_lin"))

    if dst:
        torch.save(blob, dst)
    return blob


def _digest(sd):
    """Order-independent digest of the weights, so a converted file can be tied
    back to the checkpoint it came from."""
    h = hashlib.blake2b(digest_size=16)
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--ema", action="store_true", help="convert the EMA weights")
    ap.add_argument("--bins", help="bin-edges sidecar for a categorical head")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the inferred config; write nothing")
    a = ap.parse_args(argv)
    if not a.dry_run and not a.dst:
        ap.error("dst is required unless --dry-run")

    blob = convert(a.src, None if a.dry_run else a.dst,
                   use_ema=a.ema, bins_path=a.bins)
    cfg, prov = blob["config"], blob["provenance"]
    print(f"{a.src}\n  step      {prov['step']}   weights: {prov['weights']}")
    print(f"  config    {cfg}")
    print(f"  bins      {'inlined from ' + prov['bins_file'] if blob['bins'] else 'n/a'}")
    print(f"  digest    {prov['state_digest']}")
    print(f"  verified  loads strict=True into build_fm(config)")
    if not a.dry_run:
        print(f"  -> {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
