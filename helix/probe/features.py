"""Frozen encoder features, and the random-init control.

Two models are needed, not one. ``random`` is a same-architecture, unloaded twin
— it measures what the probe HEAD can extract from features of the right shape,
which is the floor a trained model must beat to have learned anything. Without
it a positive score says only that the head has capacity.

Weights come from the checkpoint's EMA. That diverges from the reference, whose
probes all load ``ck["model"]`` (``--use_ema`` defaults to 0 and ``triangulate``
has no EMA path), so numbers here are not directly comparable with
``probe_ext.jsonl``. The EMA is the better representation on a flat WSD phase —
which is what ``mae_ddp`` saved it for — and the choice is recorded in every
results row so the two are never silently mixed.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

__all__ = ["load_probe_model", "features_at_layer", "gather_cell_features"]


def load_probe_model(checkpoint, *, random_init=False, weights="ema", device=None,
                     random_seed=0):
    """``(model, meta)`` from a converted checkpoint, trained or random-init.

    ``weights='ema'`` prefers ``state_dict_ema`` / the sidecar EMA and falls back
    to the raw weights ONLY if the checkpoint has none — loudly, in ``meta``, so
    a run cannot silently report EMA numbers it did not use.

    ``random_seed`` seeds the ``random_init`` draw. It used to be unseeded, so the
    control was a DIFFERENT network in every probe process, drawn from torch's
    process-start seed — which means two arms of an A/B were compared against two
    different nulls and nothing recorded that. The draw turns out to contribute
    little on its own (measured sigma 0.0021 at 150 events, 0.0024 at 30, 0.0005
    between two nets at 120), but "little" is not "nothing" and it cost nothing to
    fix. It is recorded in ``meta`` so a results row can carry it.
    """
    import torch
    from helix.model import build_fm

    from helix.model.checkpoint import is_export_dir, load_export_dir

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if random_init and random_seed is not None:
        # Before build_fm, which is where every parameter is drawn.
        torch.manual_seed(int(random_seed))
    if is_export_dir(checkpoint):
        # What WE train: a `pimm export` directory. bin_edges is persistent now,
        # so the edges arrive with the weights and nothing needs a sidecar.
        model, meta = load_export_dir(checkpoint, device=device)
        if random_init:
            fresh = build_fm(meta["config"])
            fresh.to(dev).eval()
            for prm in fresh.parameters():
                prm.requires_grad_(False)
            return fresh, dict(meta, weights="random-init", random_init=True,
                               random_seed=random_seed)
        # An export dir holds ONE set of weights, whatever `pimm export` was
        # pointed at, so `weights=` cannot be honoured here. Say so loudly: the
        # raw weights of a flat-LR WSD run sit at full LR noise for the entire
        # stable phase, which is why the EMA exists, and silently probing them
        # while the caller asked for the EMA compares two noisy draws rather than
        # two models. Export from model_ema.pth to probe the EMA.
        # Judge on the SOURCE path the export recorded, not on the exported
        # filename. `pimm export` always writes model.safetensors / model.bin,
        # so a filename test answers the same for an EMA export and a raw one —
        # it warned on every export including correct ones, which is how a
        # warning stops being read.
        src = meta.get("weights_source")
        is_ema = None if not src else ("ema" in os.path.basename(str(src)).lower())
        if weights == "ema" and is_ema is False:
            meta = dict(meta, warning=(
                f"requested weights='ema' but this export was written from "
                f"{src!r}, which is not an EMA checkpoint. An export dir carries "
                f"one weight set; re-export from model_ema.pth. The raw weights "
                f"of a flat-LR WSD run sit at full LR noise for the whole stable "
                f"phase — which is why the EMA exists — so probing them while "
                f"asking for the EMA compares two noisy draws, not two models."))
            warnings.warn(meta["warning"], RuntimeWarning, stacklevel=2)
        elif weights == "ema" and is_ema is None:
            meta = dict(meta, warning=(
                f"requested weights='ema' but this export records no source "
                f"checkpoint, so which weight set it holds CANNOT be determined "
                f"from the directory. Treat the result as unattributed."))
            warnings.warn(meta["warning"], RuntimeWarning, stacklevel=2)
        return model, dict(meta, requested_weights=weights, random_init=False,
                           weights_are_ema=is_ema)

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "config" not in blob or "state_dict" not in blob:
        raise ValueError(
            f"{checkpoint} is not a converted checkpoint (no config/state_dict). "
            f"Convert it with tools/convert_fm_ckpt.py, which also records the "
            f"operating point the weights were trained at.")

    cfg = dict(blob["config"])
    if isinstance(cfg.get("film"), list):
        cfg["film"] = tuple(cfg["film"])
    model = build_fm(cfg)

    used = "random-init"
    if not random_init:
        from helix.model.checkpoint import load_converted
        used = load_converted(model, blob, prefer=weights)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = dict(weights=used, requested_weights=weights,
                random_init=bool(random_init),
                random_seed=random_seed if random_init else None,
                config={k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in cfg.items()},
                tokenizer=blob.get("tokenizer"),
                bins_present=blob.get("bins") is not None)
    if not random_init and weights == "ema" and used != "ema":
        meta["warning"] = (
            "EMA requested but the checkpoint carries none; used raw weights. "
            "These numbers are NOT EMA numbers.")
    return model, meta


def features_at_layer(model, batch, layer, *, amp=True):
    """Per-cell features from one encoder layer. ``(n_cells, d)`` float32."""
    import torch

    dev = next(model.parameters()).device
    with torch.no_grad():
        if amp and dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.encode_layers(batch, {layer})[layer]
        else:
            out = model.encode_layers(batch, {layer})[layer]
    return out.float()


def gather_cell_features(feats, cell_rows):
    """Gather each row's ``n_bands`` cell features into one vector.

    ``cell_rows`` is ``(n_rows, n_bands)`` of cell INDICES, ``-1`` where that row
    has no cell in that band. Missing bands are left as zeros and their absence
    is carried separately as presence bits (see :mod:`helix.probe.patches`) —
    zero is a legitimate feature value, so absence must not be encoded only as
    one.
    """
    import torch

    cell_rows = np.asarray(cell_rows, np.int64)
    n_rows, n_bands = cell_rows.shape
    fd = feats.shape[1]
    out = torch.zeros((n_rows, n_bands * fd), device=feats.device, dtype=feats.dtype)
    for b in range(n_bands):
        cid = cell_rows[:, b]
        m = cid >= 0
        if not m.any():
            continue
        rows = torch.from_numpy(np.nonzero(m)[0]).to(feats.device)
        cols = torch.from_numpy(cid[m]).to(feats.device)
        out[rows, b * fd:(b + 1) * fd] = feats[cols]
    return out
