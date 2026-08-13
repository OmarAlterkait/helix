"""Probing the coefficient FM: does the representation carry physics?

Two probes, both predicting ``u`` — the along-wire coordinate, the degree of
freedom a single wire plane provably cannot determine — from frozen encoder
features, scored with ``fisher_r`` (per-(event, plane) Pearson-r, Fisher-z
averaged):

  ``mlp``          single-plane readout. Arms trained / random / raw / geo.
                   Answers: is ``u`` present above the geometry floor?
  ``triangulate``  adds drift-time-matched context from the other two planes.
                   Arms solo / cross / xwire. Answers: if it is weak
                   single-plane, is the information gone or just un-combined?

Two stages, because the truth is expensive to derive and cheap to reuse:

  stage 1  ``scripts/dump_probe_truth.py`` -> a per-PIXEL truth artifact beside
           the corpus. Depends on ``hits`` + ``step`` and NOTHING else — no
           corpus, no tokenizer, no checkpoint — which is why it survives a
           corpus rebuild or a change of ``PatchConfig``.
  stage 2  ``scripts/run_probe.py`` -> features from a checkpoint, joined to the
           truth through ``helix.model.tokenize.pixel_cells``.

Deliberately pimm-free: ``pimm`` is not importable in helix's test image, so
anything behind that import cannot be tested. Everything here needs only numpy,
h5py, torch and (for the corpus) ``pimm_data``, all of which are.
"""

from __future__ import annotations

_LAZY = {
    "fit_alongwire": "helix.probe.alongwire",
    "verify_alongwire": "helix.probe.alongwire",
    "u_of": "helix.probe.alongwire",
    "decode_hits_plane": "helix.probe.truth",
    "group_centroids": "helix.probe.truth",
    "pixel_truth": "helix.probe.truth",
    "fit_probe": "helix.probe.fit",
    "fisher_r": "helix.probe.metrics",
    "load_probe_model": "helix.probe.features",
    "features_at_layer": "helix.probe.features",
    "gather_cell_features": "helix.probe.features",
    "patch_rows": "helix.probe.patches",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'helix.probe' has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(target), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
