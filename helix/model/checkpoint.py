"""Reading a converted FM checkpoint's recorded operating point.

Separate from :mod:`helix.integrations.pimm` because none of this needs pimm —
it reads a blob and returns a :class:`~helix.model.tokenize.PatchConfig`. Living
in the pimm adapter made it unimportable without pimm installed, which is the
opposite of what a config recipe needs. Separate from
:mod:`helix.model.tokenize` because that module is deliberately torch-free and a
subprocess test enforces it.
"""

from __future__ import annotations

def patch_config_from_checkpoint(checkpoint):
    """The ``PatchConfig`` a converted checkpoint was TRAINED with.

    The blob records ``tokenizer`` (pw, pt, cell_t) because none of it is
    recoverable from the weights. Nothing read it: ``build_coeff_fm`` took only
    config/state_dict/bins, and every recipe built ``CoeffTokenize`` with no
    ``cfg=``, falling through to ``PatchConfig()`` — whose ``cell_t`` default is
    ``centroid`` while m113 trained on ``grid_center``. Measured on corpus event
    0, ``t_phys`` differs on 94.06% of 30976 cells, mean |delta| 19.5. So the
    encode recipe restored real trained weights and then fed them a time
    coordinate the model had never seen.

    Returns ``None`` when the checkpoint records no tokenizer, so a caller can
    tell "not recorded" from "recorded as the default".
    """
    import torch
    from helix.model.tokenize import PatchConfig

    # A `pimm export` DIRECTORY is a valid checkpoint here too. It was not
    # handled: load_probe_model learned about export dirs but this did not, so
    # probing a pimm-trained model died on `IsADirectoryError` in torch.load
    # before it reached the loader that would have coped.
    if is_export_dir(checkpoint):
        tok = _export_tokenizer_cfg(checkpoint)
        if not tok:
            return None
        kw = {k: tok[k] for k in ("pw", "pt", "cell_t") if k in tok}
        if tok.get("n_bands"):
            kw["n_bands"] = int(tok["n_bands"])
        return PatchConfig(**kw)

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tok = blob.get("tokenizer")
    if not tok:
        return None
    kw = {k: tok[k] for k in ("pw", "pt", "cell_t") if k in tok}
    if tok.get("n_bands"):
        kw["n_bands"] = int(tok["n_bands"])
    return PatchConfig(**kw)


def _export_tokenizer_cfg(path):
    """``CoeffTokenize``'s ``cfg`` dict from a ``pimm export`` config.json, or None.

    Shared by :func:`patch_config_from_checkpoint` and :func:`load_export_dir` so
    the two cannot disagree about where the tokenizer geometry lives. It travels
    in the transform list rather than the model section, because it describes how
    coefficients become tokens, not the architecture.
    """
    import json
    import os

    cfg_path = next((os.path.join(path, c) for c in _EXPORT_CONFIGS
                     if os.path.exists(os.path.join(path, c))), None)
    if cfg_path is None:
        return None
    full = json.load(open(cfg_path))
    for t in (full.get("transform") or []):
        if isinstance(t, dict) and t.get("type") == "CoeffTokenize":
            return dict(t.get("cfg") or {})
    return None


def load_converted(model, blob, *, prefer="raw"):
    """Load a converted checkpoint's weights into ``model``, strict.

    Exists so the bins migration lives in ONE place. ``bin_edges`` is a
    persistent buffer now and rides inside the state_dict, but blobs converted
    before that keep the edges beside the weights under ``blob["bins"]`` — so a
    strict load of an old blob would fail on a missing key. Returns which
    weights were used.
    """
    import torch

    sd = blob["state_dict"]
    used = "raw"
    if prefer == "ema":
        ema = blob.get("state_dict_ema") or blob.get("ema")
        if ema:
            sd, used = ema, "ema"
    if blob.get("bins") and "bin_edges" not in sd:
        sd = dict(sd)
        sd["bin_edges"] = torch.as_tensor(blob["bins"]["edges"])
        # The converted blob may also carry the centroids; take them when
        # present, and let the backfill below supply NaN when it does not.
        for _n, _k in (("bin_cent_asinh", "cent_asinh"),
                       ("bin_cent_ratio", "cent_ratio")):
            if blob["bins"].get(_k) is not None:
                sd[_n] = torch.as_tensor(blob["bins"][_k])
    sd = _backfill_centroids(model, sd)
    model.load_state_dict(sd, strict=True)
    return used


#: Bin-centroid buffers, and how to derive each from the edges. Persistent since
#: the categorical read-back was fixed, so any checkpoint written before that
#: carries `bin_edges` without them.
#:
#: `bin_cent_lin` is not here: it was E[raw ADC | bin] pooled across planes with
#: 22%-different sigmas, nothing read it, and it is gone. A checkpoint that still
#: carries the key loads through `_drop_stale` below.
_CENT_BUFFERS = ("bin_cent_asinh", "bin_cent_ratio")

#: Buffers removed from the model that an older state_dict may still carry.
#: Dropping them is what keeps `strict=True` an honest check on the WEIGHTS.
_STALE_BUFFERS = ("bin_cent_lin",)


def apply_bins(model, bins, *, log=None):
    """Install a bin sidecar onto ``model``. The ONLY caller of ``set_bins``.

    Every path that has a sidecar in hand — the trainer, the converter, the
    golden capture — goes through here, so "how do bins reach a model" has one
    answer. The direct calls it replaces passed centroids POSITIONALLY, and when
    the signature grew a table one of them silently bound its argument to the
    wrong slot; the metrics that depended on it then read NaN and reported
    nothing. ``set_bins`` is keyword-only now, but the deeper fix is that there
    is one call.

    ``bins`` is the mapping ``derive_coeff_bins`` writes: ``edges`` required,
    ``cent_asinh``/``cent_ratio`` optional (``set_bins`` derives what is absent).
    Any other key — ``cent_lin``, provenance — is ignored rather than rejected,
    so an old sidecar still applies.

    When the model already holds finite edges, logs the largest disagreement
    before overwriting: silently replacing a checkpoint's own trained edges with
    a sidecar's is how a model gets evaluated against a grid it never saw.
    """
    import torch

    assert "edges" in bins, f"bin sidecar has no 'edges' (keys: {sorted(bins)})"
    prev = getattr(model, "bin_edges", None)
    if prev is not None and torch.isfinite(prev).all():
        d = (prev.detach().cpu() - torch.as_tensor(bins["edges"],
                                                   dtype=prev.dtype)).abs().max()
        msg = (f"apply_bins: overwriting existing bin_edges, max|delta| = {d:.3e}")
        if d > 0 and log is not None:
            log(msg)
        elif d > 0:
            import warnings
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return model.set_bins(bins["edges"],
                          cent_asinh=bins.get("cent_asinh"),
                          cent_ratio=bins.get("cent_ratio"))


def _backfill_centroids(model, sd):
    """Derive any centroid buffer the checkpoint predates, from its own edges.

    Keeps ``strict=True`` meaningful — a missing WEIGHT stays an error — while
    letting pre-fix checkpoints load. Deriving from ``sd["bin_edges"]`` rather
    than copying the model's freshly-constructed NaN is what makes the loaded
    model obey the same invariant ``set_bins`` enforces: no centroid table is
    ever NaN, so no consumer has to test for one. A derived cent_ratio under-reads
    sum|centroid| by 2.7-3.0% per band against a measured one (outer bins ~24%
    low); that is the documented cost of a checkpoint that never stored them.
    """
    import torch
    from helix.model.tokenize import bin_centroids_asinh, bin_centroids_ratio

    sd = {k: v for k, v in sd.items() if k not in _STALE_BUFFERS}
    missing = [n for n in _CENT_BUFFERS if n not in sd and hasattr(model, n)]
    if missing and "bin_edges" in sd:
        e = torch.as_tensor(sd["bin_edges"]).detach().cpu().numpy()
        for n, fn in (("bin_cent_asinh", bin_centroids_asinh),
                      ("bin_cent_ratio", bin_centroids_ratio)):
            if n in missing:
                sd[n] = torch.as_tensor(fn(e), dtype=torch.as_tensor(sd["bin_edges"]).dtype)
    elif missing:
        for n in missing:                      # n_bins=0: buffers are absent anyway
            sd[n] = getattr(model, n).detach().clone()
    if "bin_cent_measured" not in sd and hasattr(model, "bin_cent_measured"):
        sd["bin_cent_measured"] = torch.zeros(2, dtype=torch.uint8)
    return sd


#: Filenames ``pimm export`` writes, in preference order.
_EXPORT_CONFIGS = ("config.json", "training_config.json")
_EXPORT_WEIGHTS = ("model.safetensors", "model.bin")


def is_export_dir(path):
    """True if ``path`` looks like a ``pimm export`` directory."""
    import os
    return os.path.isdir(path) and any(
        os.path.exists(os.path.join(path, w)) for w in _EXPORT_WEIGHTS)


def load_export_dir(path, *, device=None):
    """``(model, meta)`` from a ``pimm export`` directory.

    This is the forward path for anything WE train. ``pimm export`` already
    writes the HuggingFace-shaped pair — weights plus the resolved config beside
    them — so there is no helix-specific checkpoint format to invent, and the
    directory is portable by construction.

    ``tools/convert_fm_ckpt.py`` stays frozen as the one-time rescue of the
    historical m113 checkpoint, which could not describe itself. Nothing trained
    from here should go through it.
    """
    import json
    import os
    import torch
    from helix.model import build_fm
    from helix.model.tokenize import PatchConfig

    cfg_path = next((os.path.join(path, c) for c in _EXPORT_CONFIGS
                     if os.path.exists(os.path.join(path, c))), None)
    if cfg_path is None:
        raise ValueError(
            f"{path} has weights but none of {_EXPORT_CONFIGS} — the "
            f"architecture is not recoverable. Re-export with the run's config.")
    full = json.load(open(cfg_path))
    mcfg = dict(full.get("model") or {})
    if not mcfg:
        raise ValueError(f"{cfg_path} carries no 'model' section")
    for k in ("type", "checkpoint", "bins", "weights"):
        mcfg.pop(k, None)                      # builder selectors, not arch
    if isinstance(mcfg.get("film"), list):
        mcfg["film"] = tuple(mcfg["film"])

    model = build_fm(mcfg)
    wpath = next(os.path.join(path, w) for w in _EXPORT_WEIGHTS
                 if os.path.exists(os.path.join(path, w)))
    if wpath.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise SystemExit(
                f"{wpath} needs the safetensors package, which is absent here. "
                f"Re-export with --no-safe-serialization to get model.bin, or "
                f"install safetensors in this image.")
        sd = load_file(wpath)
    else:
        sd = torch.load(wpath, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    # The bin CENTROID buffers became persistent when the categorical read-back
    # was fixed, so an export written before that carries `bin_edges` but not
    # them. Backfill from the model's own NaN-initialised buffers rather than
    # relaxing `strict`: a missing WEIGHT must still be an error, and NaN is
    # exactly the "absent" signal `bin_centroids_ratio` already falls back on.
    sd = _backfill_centroids(model, sd)
    model.load_state_dict(sd, strict=True)     # bin_edges rides along, persistent

    # Tokenizer geometry travels in the same config, inside the transform list.
    tok = _export_tokenizer_cfg(path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    meta = dict(source="pimm-export", weights=os.path.basename(wpath),
                config=mcfg, tokenizer=tok,
                patch_config=PatchConfig(**tok) if tok else None)
    return model, meta
