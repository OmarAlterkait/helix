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

from dataclasses import replace
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
    from helix.model.artifact import build, load

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if random_init and random_seed is not None:
        # Before build_fm, which is where every parameter is drawn.
        torch.manual_seed(int(random_seed))

    art = load(checkpoint)
    used = "random-init" if random_init else art.pick(weights)[1]
    if random_init:
        art = replace(art, state_dict=None, state_dict_ema=None)
    model = build(art, prefer=weights, device=dev)

    meta = dict(source=art.fmt, weights=used, requested_weights=weights,
                random_init=bool(random_init),
                random_seed=random_seed if random_init else None,
                config={k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in art.arch.items()},
                tokenizer=_tokenizer_meta(art.op),
                bins_present=art.op.bins is not None,
                weights_source=art.weights_source,
                weights_are_ema=_are_ema(art),
                # What the artifact says about itself: the corpus basis_digest
                # it trained on, the helix commit, the weight content hash. A
                # `pimm export` carries none of it — see scripts/export_artifact.py.
                provenance=dict(art.provenance))
    if not random_init:
        w = _weights_warning(art, weights, used)
        if w:
            meta["warning"] = w
            if art.fmt == "pimm-export":
                warnings.warn(w, RuntimeWarning, stacklevel=2)
    return model, meta


def _are_ema(art):
    """Tri-state: True, False, or None for "the source cannot say".

    Judge on what the artifact RECORDS, never on the exported filename.
    ``pimm export`` always writes model.safetensors / model.bin whatever it was
    exported from, so a filename test answers the same for an EMA export and a
    raw one — it warned on every export including correct ones, which is how a
    warning stops being read. And pimm's ``_sanitize_config`` nulls the one key
    that could have recorded the source, so a real export is always None.
    """
    if art.state_dict_ema:
        return True                     # a converted blob carries both sets
    return None if art.weights == "unknown" else art.weights == "ema"


def _tokenizer_meta(op):
    """The recorded tokenizer block, or None when nothing was recorded."""
    if not op.recorded:
        return None
    return {k: v for k, v in (("pw", op.pw), ("pt", op.pt),
                              ("cell_t", op.cell_t), ("n_bands", op.n_bands))
            if v is not None}


def _weights_warning(art, requested, used):
    """Why the weight set in hand may not be the one the caller asked for.

    An export dir holds ONE set, so ``weights=`` cannot be honoured there. Say
    so loudly: the raw weights of a flat-LR WSD run sit at full LR noise for the
    entire stable phase, which is why the EMA exists, and silently probing them
    while the caller asked for the EMA compares two noisy draws rather than two
    models.
    """
    if requested != "ema" or used == "ema":
        return None
    if art.fmt == "pimm-export" and art.weights_source:
        return (f"requested weights='ema' but this export was written from "
                f"{art.weights_source!r}, which is not an EMA checkpoint. An "
                f"export dir carries one weight set; re-export from "
                f"model_ema.pth. The raw weights of a flat-LR WSD run sit at "
                f"full LR noise for the whole stable phase — which is why the "
                f"EMA exists — so probing them while asking for the EMA "
                f"compares two noisy draws, not two models.")
    if used == "unknown":
        return (f"requested weights='ema' but {art.source} records no source "
                f"checkpoint, so which weight set it holds CANNOT be determined "
                f"from the directory. Treat the result as unattributed.")
    return ("EMA requested but the checkpoint carries none; used raw weights. "
            "These numbers are NOT EMA numbers.")


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
