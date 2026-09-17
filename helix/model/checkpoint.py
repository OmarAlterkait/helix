"""Applying what was read: weights in, bins on, PatchConfig out.

:mod:`helix.model.artifact` decides WHAT a checkpoint is and reads it; this
module applies what it read — the strict state_dict load with its migrations,
the bin sidecar, and the ``PatchConfig`` a recipe needs. The split is
deliberate: reading must not require a model, and building must not have to
know about file formats.

NOTHING HERE READS A FILE FORMAT. It briefly did: `is_export_dir`,
`load_export_dir` and `_export_tokenizer_cfg` stayed behind when the one-owner
move landed, along with a `_patch_config_from_tok` that no longer had callers
and an `_EXPORT_CONFIGS`/`_EXPORT_WEIGHTS` re-export whose comment claimed
"recipes and tests import them from here" when nothing did. Two modules that
both know what an export directory looks like is the exact condition
`artifact.py` exists to end, so the second one is gone -- use
``artifact.load(path)`` and ``artifact.build(art)``.

Separate from :mod:`helix.integrations.pimm` because none of it needs pimm.
Living in the pimm adapter made it unimportable without pimm installed, which is
the opposite of what a config recipe needs. Separate from
:mod:`helix.model.tokenize` because that module is deliberately torch-free and a
subprocess test enforces it.
"""

from __future__ import annotations

import os

def patch_config_from_checkpoint(checkpoint, cell_t=None):
    """The ``PatchConfig`` a checkpoint was TRAINED with.

    The checkpoint records pw, pt and cell_t because none of it is recoverable
    from the weights. Nothing read it: ``build_coeff_fm`` took only
    config/state_dict/bins, and every recipe built ``CoeffTokenize`` with no
    ``cfg=``, falling through to a then-defaulted ``PatchConfig()`` — whose
    ``cell_t`` was ``centroid`` while m113 trained on ``grid_center``. That
    fallback is gone (``cell_t`` is required now); this records why it had to be. Measured on corpus event
    0, ``t_phys`` differs on 94.06% of 30976 cells, mean |delta| 19.5. So the
    encode recipe restored real trained weights and then fed them a time
    coordinate the model had never seen.

    Returns ``None`` when the checkpoint records no tokenizer, so a caller can
    tell "not recorded" from "recorded as the default".

    ``cell_t`` fills the field in for a checkpoint that does not record it, and is
    CROSS-CHECKED against one that does — see :func:`patch_config`.
    """
    from helix.model.artifact import inspect

    # A `pimm export` DIRECTORY is a valid checkpoint here too. It was not
    # handled: load_probe_model learned about export dirs but this did not, so
    # probing a pimm-trained model died on `IsADirectoryError` in torch.load
    # before it reached the loader that would have coped. There is now one
    # reader for both, so the two cannot drift apart again.
    return patch_config(inspect(checkpoint).op, cell_t, checkpoint)


def patch_config(op, cell_t=None, where=""):
    """Build a :class:`PatchConfig` from an
    :class:`~helix.model.artifact.OperatingPoint`.

    Three cases callers must not conflate:

    * **nothing recorded** -> ``None``. The caller asks the user.
    * **recorded without cell_t** -> filled from ``cell_t``, or refused. The
      recorded ``pw``/``pt`` are real information, so returning ``None`` here
      would be wrong twice over: it would discard them AND silently substitute
      the ``PatchConfig`` defaults in their place.
    * **recorded, plus a conflicting ``cell_t``** -> refused. Letting the file
      quietly win is the same failure this function exists to prevent, only
      pointed the other way; the four probe scripts used to do exactly that.
    """
    if not op.recorded:
        return None
    from helix.model.tokenize import PatchConfig
    kw = {k: getattr(op, k) for k in ("pw", "pt") if getattr(op, k) is not None}
    if op.n_bands:
        kw["n_bands"] = int(op.n_bands)
    if op.cell_t is None:
        if cell_t is None:
            raise ValueError(
                f"{where} records a tokenizer (pw={kw.get('pw')}, pt={kw.get('pt')}) "
                "but no cell_t, so the time coordinate it trained on is unknown. "
                "Supply it (--cell-t on the probe scripts).")
        kw["cell_t"] = cell_t
    else:
        if cell_t is not None and cell_t != op.cell_t:
            raise ValueError(
                f"{where} records cell_t={op.cell_t!r}, but cell_t={cell_t!r} was "
                "requested. Refusing to override: one of the two is wrong, and "
                "picking either silently is how a model gets scored on a "
                "coordinate it never saw.")
        kw["cell_t"] = op.cell_t
    return PatchConfig(**kw)


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
#: `bin_cent_measured` was a 2-byte provenance buffer recording whether each
#: centroid table was measured or derived. NOTHING ever read it — the same
#: mistake as `bin_cent_lin` above, committed while removing that one. Checkpoints
#: written between then and now carry it, including the run training right now,
#: so it is dropped here rather than being a reason to keep the buffer.
_STALE_BUFFERS = ("bin_cent_lin", "bin_cent_measured")


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
    return sd


def load_state_dict(model, sd, *, bins=None):
    """Strict load, after the migrations every caller needs and forgets.

    Strips the DDP ``module.`` prefix; injects a pre-buffer checkpoint's bin
    edges from the sidecar beside it; drops buffers the model no longer has; and
    derives the bin-centroid buffers a pre-fix checkpoint predates — from the
    checkpoint's OWN edges, so the loaded model obeys the same invariant
    ``set_bins`` enforces and no consumer has to test for NaN.

    ``strict=True`` stays an honest check on the WEIGHTS: that is the whole
    reason these three migrations are explicit rather than a relaxed load.
    """
    import torch

    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    # `bin_edges` is a persistent buffer now and rides inside the state_dict,
    # but blobs converted before that keep the edges beside the weights — so a
    # strict load of an old blob would fail on a missing key.
    if bins and "bin_edges" not in sd:
        sd["bin_edges"] = torch.as_tensor(bins["edges"])
        for _n, _k in (("bin_cent_asinh", "cent_asinh"),
                       ("bin_cent_ratio", "cent_ratio")):
            if bins.get(_k) is not None:
                sd[_n] = torch.as_tensor(bins[_k])
    model.load_state_dict(_backfill_centroids(model, sd), strict=True)
    return model
