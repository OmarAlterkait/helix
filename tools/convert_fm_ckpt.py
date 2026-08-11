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
              "mup", "d_base", "ffn_mult", "wire_rope", "n_slot", "n_bins",
              "pw", "pt")

# Fields that change what the model computes but leave NO trace in the weights
# and are NOT saved by the research trainer. They live only in the run's YAML.
# `rope_split` reroutes RoPE on every serial layer; `cellt` changes the time
# coordinate the tokenizer emits. A checkpoint separated from its YAML is
# therefore evaluable only by guessing, and the guess is silent — which is what
# happened: m113 trained with rope_split=0 and cellt=canonical, and every
# evaluation of it ran at rope_split=True with a centroid tokenizer.
#
# Values are the RESEARCH argparse defaults (mae_ddp.py), used only to fill in
# what a YAML omits, never to paper over a YAML that is missing entirely.
_RESEARCH_DEFAULTS = {"rope_split": 1, "cellt": "canonical", "serial": 0,
                      "gp": 1024, "gd": 2048}
# Recorded for provenance; they shaped the weights but do not affect a forward
# pass, so they are informational rather than part of `config`.
_TRAIN_KEYS = ("mask", "mask_mode", "plane_frac", "n_planes", "ema", "lr",
               "lr_mode", "warmup", "wd", "seed", "steps", "events")


def read_train_config(path):
    """Parse a research run YAML into a plain dict.

    The files are flat ``key: value`` scalars, so the fallback parser is exact
    for them; PyYAML is used when present so anything richer still works."""
    text = open(path).read()
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = (x.strip() for x in line.split(":", 1))
        if not v:
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v.strip("'\"")
    return out


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


def convert(src, dst=None, *, use_ema=False, bins_path=None, verify=True,
            train_config=None, serial=None, rope_split=None):
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

    # ---- the operating point the weights were trained at -------------------
    yml = read_train_config(train_config) if train_config else {}
    tc = {k: yml.get(k, d) for k, d in _RESEARCH_DEFAULTS.items()}

    # A serial checkpoint and a full-attention one have IDENTICAL parameter
    # shapes — verified: the m113 state dict loads strict=True into either. So
    # nothing in the weights can settle this, and a default would just pick one
    # silently. Require the YAML, or an explicit assertion from the caller.
    if train_config:
        is_serial = bool(tc["serial"])
    elif serial is not None:
        is_serial = bool(serial)
    else:
        raise SystemExit(
            "cannot tell whether this is a serial or full-attention checkpoint, "
            "and the weights cannot settle it — the parameter shapes are the "
            "same either way, so the wrong choice loads cleanly and evaluates a "
            "different model.\n"
            "Pass --train-config <run>.yaml (which also carries rope_split and "
            "cellt, neither of which is stored in the checkpoint), or assert it "
            "with --serial/--no-serial.")
    if is_serial:
        if train_config is None and rope_split is None:
            raise SystemExit(
                "serial checkpoint, but `rope_split` is unknown. It reroutes "
                "RoPE on every serial layer, so the wrong value silently "
                "evaluates a different model (research defaults to 1; m113 "
                "trained with 0). Pass --train-config or --rope-split 0|1.")
        cfg["rope_split"] = bool(tc["rope_split"] if rope_split is None
                                 else rope_split)
        cfg["gp"], cfg["gd"] = int(tc["gp"]), int(tc["gd"])
    # Record it, so `build_fm(blob["config"])` alone rebuilds the right class and
    # no caller has to remember to pass serial= on the side.
    cfg["serial"] = is_serial

    # Tokenizer geometry. pw/pt come from the checkpoint (authoritative, they
    # were saved); cell_t comes from the YAML because it never was.
    # helix renamed research's cell-time modes; the formulas are identical.
    #   research "canonical" == helix "grid_center"  (cell_tb*pt + pt/2, then
    #                                                 (+delta)*dec - toff)
    #   research "centroid"  == helix "centroid"     (amplitude-weighted mean)
    # research/vit_tpc.py:107-117 is the definition on that side.
    _CELLT = {"canonical": "grid_center", "centroid": "centroid"}
    if str(tc["cellt"]) not in _CELLT:
        raise SystemExit(f"unknown cellt {tc['cellt']!r}; expected one of "
                         f"{sorted(_CELLT)}")
    tokenizer = {"pw": int(meta.get("pw", 16)), "pt": int(meta.get("pt", 8)),
                 "cell_t": _CELLT[str(tc["cellt"])], "n_bands": cfg.get("n_band"),
                 "cellt_research": str(tc["cellt"])}
    cfg.pop("pw", None); cfg.pop("pt", None)     # not build_fm kwargs

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
        "tokenizer": tokenizer,
        "state_dict": sd,
        "bins": bins,
        "provenance": {
            "train_config": os.path.abspath(train_config) if train_config else None,
            "train": {k: yml[k] for k in _TRAIN_KEYS if k in yml},
            "source": os.path.abspath(src),
            "weights": which,
            "step": int(ck.get("step", -1)),
            "bins_file": os.path.abspath(bins_path) if bins_path else None,
            "state_digest": _digest(sd),
        },
    }

    if verify:
        from helix.model import build_fm
        model = build_fm(cfg, serial=is_serial)
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
    ap.add_argument("--train-config", metavar="YAML",
                    help="the run's YAML — the only record of rope_split/cellt")
    ap.add_argument("--serial", dest="serial", action="store_true", default=None,
                    help="assert a serial checkpoint (needs --rope-split too)")
    ap.add_argument("--no-serial", dest="serial", action="store_false",
                    help="assert a full-attention checkpoint")
    ap.add_argument("--rope-split", type=int, choices=(0, 1),
                    help="serial RoPE routing, when there is no --train-config")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the inferred config; write nothing")
    a = ap.parse_args(argv)
    if not a.dry_run and not a.dst:
        ap.error("dst is required unless --dry-run")

    blob = convert(a.src, None if a.dry_run else a.dst,
                   use_ema=a.ema, bins_path=a.bins,
                   train_config=a.train_config, serial=a.serial,
                   rope_split=a.rope_split)
    cfg, prov = blob["config"], blob["provenance"]
    print(f"{a.src}\n  step      {prov['step']}   weights: {prov['weights']}")
    print(f"  config    {cfg}")
    print(f"  tokenizer {blob['tokenizer']}")
    if prov.get("train"):
        print(f"  train     {prov['train']}")
    print(f"  bins      {'inlined from ' + prov['bins_file'] if blob['bins'] else 'n/a'}")
    print(f"  digest    {prov['state_digest']}")
    print(f"  verified  loads strict=True into build_fm(config)")
    if not a.dry_run:
        print(f"  -> {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
