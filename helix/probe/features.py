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

import numpy as np

__all__ = ["load_probe_model", "features_at_layer", "gather_cell_features"]


def load_probe_model(checkpoint, *, random_init=False, weights="ema", device=None):
    """``(model, meta)`` from a converted checkpoint, trained or random-init.

    ``weights='ema'`` prefers ``state_dict_ema`` / the sidecar EMA and falls back
    to the raw weights ONLY if the checkpoint has none — loudly, in ``meta``, so
    a run cannot silently report EMA numbers it did not use.
    """
    import torch
    from helix.model import build_fm

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
        sd, used = blob["state_dict"], "raw"
        if weights == "ema":
            ema = blob.get("state_dict_ema") or blob.get("ema")
            if ema:
                sd, used = ema, "ema"
        model.load_state_dict(sd, strict=True)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = dict(weights=used, requested_weights=weights,
                random_init=bool(random_init),
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
