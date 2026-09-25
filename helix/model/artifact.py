"""What is this model, and against what number may I compare it?

ONE place that knows what a checkpoint is. There were eight shapes in
circulation, read by eight independent ``torch.load`` sites that each knew a
subset, and the scar is recorded in :mod:`helix.model.checkpoint`: "a ``pimm
export`` DIRECTORY is a valid checkpoint here too. It was not handled:
load_probe_model learned about export dirs but this did not, so probing a
pimm-trained model died on IsADirectoryError". One loader learned a format and
its sibling did not, and nothing could have caught that because no test named
the set of formats.

THE SPLIT IS BY PURPOSE, NOT BY FILE FORMAT.

  RESUME state -- weights plus optimizer, scheduler, RNG, sampler and step. It
  exists so training continues bit-identically, it is pimm's, and NOTHING here
  reads it: ``<save_path>/model/last/`` (DCP) and ``iter_N.pth`` belong to the
  trainer and to the chain launcher's floor logic.

  EVAL state -- one weight set, the architecture, the operating point, and
  provenance. It exists so a NUMBER is reproducible and attributable. That is
  what this module reads, and what :func:`save` writes.

Conflating the two is how the mess grew. ``model_ema.pth`` is written by a
resume hook (the shadow must keep accumulating across preemptions) and read by
inference (it is the weight set you score), and it serves the second badly: it
records no architecture, no tokenizer and no corpus, so a raw pimm checkpoint
cannot be scored at all -- which is the error message four call sites had each
grown their own copy of.

INSPECTION IS SEPARATE FROM INSTANTIATION because most callers do not want a
model. ``patch_config_from_checkpoint`` wants the tokenizer and ``_load_bins``
wants the edges; both used to load every tensor in the file to read a handful of
scalars.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

#: What ``pimm export`` writes, in preference order.
_EXPORT_WEIGHTS = ("model.safetensors", "model.bin")
_EXPORT_CONFIGS = ("config.json", "training_config.json")
#: What :func:`save` writes.
_ARTIFACT_JSON = "artifact.json"
_ARTIFACT_WEIGHTS = "weights.safetensors"

#: Every shape :func:`detect` distinguishes. The format-matrix test iterates it,
#: so adding a shape without teaching the matrix about it fails.
FORMATS = ("helix-eval", "pimm-export", "converted", "research",
           "dcp-resume", "raw-state-dict", "unknown-dir", "unknown-file")
#: The shapes that carry enough to score a number. The rest are refused BY NAME:
#: a refusal that says which shape it got and what to run instead is worth far
#: more than a generic one, and costs only the dict entry.
#:
#: `converted` and `research` were readable until the converter was retired.
#: They are still DETECTED, because both still exist on disk -- m113's blob sits
#: in the archive as its own lineage -- and "unrecognised checkpoint shape" would
#: be a hostile thing to tell someone who found one.
READABLE = ("helix-eval", "pimm-export")


@dataclass(frozen=True)
class OperatingPoint:
    """Everything that moves the NUMBER, and nothing that does not.

    Not the training config: data roots, save paths, hook lists and optimizer
    settings are resume or site concerns. This is the subset a second machine
    needs to reproduce a score, and the one thing no existing artifact carries
    whole -- ``cell_t`` alone moves 94.06% of cells (mean |delta| 19.5 ticks),
    and getting it wrong yields a plausible number rather than an error.

    ``bins`` is inlined rather than referenced because a sidecar path is exactly
    the kind of thing that does not survive a move between filesystems. ``None``
    means the edges ride in the state_dict as the persistent ``bin_edges``
    buffer, which is the case for everything trained since.
    """
    cell_t: str | None = None
    pw: int | None = None
    pt: int | None = None
    n_bands: int | None = None
    bins: dict | None = None

    @property
    def recorded(self):
        """False when the source recorded no tokenizer at all.

        Distinguishes "not recorded" from "recorded as the default" -- the
        distinction that the encode recipe's silent ``PatchConfig()`` fallback
        erased.
        """
        return any(v is not None for v in (self.cell_t, self.pw, self.pt))


@dataclass(frozen=True)
class Artifact:
    """One weight set plus everything needed to reproduce a number from it."""
    arch: dict                        # build_fm(**arch)
    op: OperatingPoint
    #: "ema" | "raw" | "unknown". Tri-state on purpose: a `pimm export` records
    #: NOTHING about which checkpoint it consumed (`_sanitize_config` nulls
    #: `weight`), so every real export is "unknown", and reporting that as "raw"
    #: would let a probe compare an EMA arm against a raw one in silence.
    weights: str = "unknown"
    weights_source: str | None = None
    provenance: dict = field(default_factory=dict)
    source: str = ""                  # the path it was read from
    fmt: str = ""                     # which of FORMATS it was
    state_dict: dict | None = None    # None when only inspected


def detect(path):
    """Which of :data:`FORMATS` ``path`` is. Never guesses beyond them."""
    path = str(path)
    # Checked here rather than left to torch: a path that does not exist is the
    # most common way to get this wrong (a typo, or a $LSCRATCH artifact written
    # on a different node), and torch reports it from four frames down inside
    # serialization.py, which reads as though the checkpoint were malformed.
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {path}")
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, _ARTIFACT_JSON)):
            return "helix-eval"
        if any(os.path.exists(os.path.join(path, w)) for w in _EXPORT_WEIGHTS):
            return "pimm-export"
        # Recognised so the refusal can SAY what it is. A DCP directory used to
        # reach torch.load and die on IsADirectoryError.
        if (os.path.exists(os.path.join(path, ".metadata"))
                or any(f.startswith("__") and f.endswith(".distcp")
                       for f in os.listdir(path))):
            return "dcp-resume"
        return "unknown-dir"
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    keys = set(blob) if isinstance(blob, dict) else set()
    if {"config", "state_dict"} <= keys:
        return "converted"
    if keys & {"model", "ema"}:
        return "research"
    return "raw-state-dict" if "state_dict" in keys else "unknown-file"


#: The one copy of the message four call sites had each grown their own of.
_EXPORT_THE_RUN = (
    "Export the RUN, which has the architecture and the tokenizer beside the "
    "weights, then promote it so it can say which weight set it holds:\n"
    "    pimm export --run-dir <save_path> model_ema.pth <tmp_dir>\n"
    "    scripts/export_artifact.py <tmp_dir> --weights ema --corpus <corpus> "
    "-o <artifact_dir>\n"
    "then pass <artifact_dir>.")

#: Where m113 lives now. Named rather than described because it is the single
#: reason anyone still meets a `converted` blob.
_M113 = "$HELIX_ARCHIVE/fm_m113_artifact"


def _refuse(path, fmt):
    if fmt == "dcp-resume":
        raise ValueError(
            f"{path} is RESUME state (a distributed-checkpoint directory): "
            f"weights plus optimizer, scheduler, RNG and sampler. It exists to "
            f"continue training, not to be scored, and it records neither the "
            f"architecture nor the tokenizer.\n{_EXPORT_THE_RUN}")
    if fmt == "raw-state-dict":
        raise ValueError(
            f"{path} is a raw checkpoint: it holds weights and nothing else, so "
            f"neither the architecture nor the tokenizer the weights were "
            f"trained with is recoverable from it.\n{_EXPORT_THE_RUN}")
    if fmt in ("converted", "research"):
        raise ValueError(
            f"{path} is a {fmt} checkpoint. tools/convert_fm_ckpt.py, the only "
            f"thing that could produce or read one, was retired once m113 -- its "
            f"sole subject -- was promoted to a self-describing eval artifact.\n"
            f"If you want m113, it is at {_M113}: same weights (digest "
            f"7d795cc3ab49f90a79f98927647028b5), plus the operating point, the "
            f"pre-tau basis_digest it trained on, and this blob's whole "
            f"provenance carried across.\n"
            f"If you want something else, the converter is in git history; "
            f"reviving it to read one blob is almost certainly the wrong trade "
            f"against re-exporting the run.")
    raise ValueError(
        f"{path}: unrecognised checkpoint shape ({fmt}). Expected one of "
        f"{READABLE}.")


def inspect(path):
    """Architecture, operating point and provenance WITHOUT loading weights."""
    return _read(path, weights=False)


def load(path):
    """As :func:`inspect`, plus ``state_dict``."""
    return _read(path, weights=True)


def _read(path, *, weights):
    path = str(path)
    fmt = detect(path)
    if fmt not in READABLE:
        _refuse(path, fmt)
    if fmt == "helix-eval":
        return _read_helix_eval(path, weights)
    return _read_pimm_export(path, weights)


#: Tokenizer keys the operating point models. Anything else a source recorded is
#: kept as provenance rather than dropped -- `cellt_research` is the research-side
#: name for m113's cell_t, and it is the only surviving record of that mapping.
_OP_KEYS = ("cell_t", "pw", "pt", "n_bands")


def _op(tok, *, n_bands=None, bins=None):
    tok = dict(tok or {})
    n = tok.get("n_bands", n_bands)
    return OperatingPoint(cell_t=tok.get("cell_t"), pw=tok.get("pw"),
                          pt=tok.get("pt"),
                          n_bands=None if n is None else int(n), bins=bins)


def _tok_extra(tok):
    return {k: v for k, v in dict(tok or {}).items() if k not in _OP_KEYS}


def _read_helix_eval(path, weights):
    meta = json.load(open(os.path.join(path, _ARTIFACT_JSON)))
    sd = None
    if weights:
        from safetensors.torch import load_file
        sd = load_file(os.path.join(path, _ARTIFACT_WEIGHTS))
    arch = dict(meta["arch"])
    # JSON has no tuple. `film` is one, and build_fm's behaviour depends on it,
    # so it is restored here exactly as the other two readers restore it.
    if isinstance(arch.get("film"), list):
        arch["film"] = tuple(arch["film"])
    return Artifact(arch=arch,
                    op=OperatingPoint(**meta["operating_point"]),
                    weights=meta.get("weights", "unknown"),
                    weights_source=meta.get("weights_source"),
                    provenance=dict(meta.get("provenance") or {}),
                    source=path, fmt="helix-eval", state_dict=sd)


def export_tokenizer_cfg(path):
    """``CoeffTokenize``'s ``cfg`` from a ``pimm export`` config.json, or None.

    The tokenizer geometry travels in the transform list rather than the model
    section, because it describes how coefficients become tokens, not the
    architecture.

    The TRAIN split's transform is read first: it is what the weights saw. The
    top-level ``transform`` is only a module variable the base config builds
    the splits from, so a derived config that restates the splits (a patch-size
    variant) leaves it stale -- reading it scored a pw=32 model at pw=16.
    """
    cfg_path = _export_config_path(path)
    if cfg_path is None:
        return None
    full = json.load(open(cfg_path))
    train = ((full.get("data") or {}).get("train") or {}).get("transform")
    for t in (train or full.get("transform") or []):
        if isinstance(t, dict) and t.get("type") == "CoeffTokenize":
            return dict(t.get("cfg") or {})
    return None


def _export_config_path(path):
    return next((os.path.join(path, c) for c in _EXPORT_CONFIGS
                 if os.path.exists(os.path.join(path, c))), None)


def _read_pimm_export(path, weights):
    cfg_path = _export_config_path(path)
    if cfg_path is None:
        raise ValueError(
            f"{path} has weights but none of {_EXPORT_CONFIGS} — the "
            f"architecture is not recoverable. Re-export with the run's config.")
    full = json.load(open(cfg_path))
    arch = dict(full.get("model") or {})
    if not arch:
        raise ValueError(f"{cfg_path} carries no 'model' section")
    n_band = arch.get("n_band")
    for k in ("type", "checkpoint", "bins", "weights"):
        arch.pop(k, None)                 # builder selectors, not architecture
    if isinstance(arch.get("film"), list):
        arch["film"] = tuple(arch["film"])

    sd = None
    if weights:
        wpath = next(os.path.join(path, w) for w in _EXPORT_WEIGHTS
                     if os.path.exists(os.path.join(path, w)))
        if wpath.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError:
                raise SystemExit(
                    f"{wpath} needs the safetensors package, which is absent "
                    f"here. Re-export with --no-safe-serialization to get "
                    f"model.bin, or install safetensors in this image.")
            sd = load_file(wpath)
        else:
            import torch
            sd = torch.load(wpath, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    # WHICH checkpoint these tensors came from. The exported file is always
    # named model.safetensors / model.bin whatever it was exported FROM, so the
    # filename carries no provenance. The resolved config's `weight` is the only
    # record — and pimm's _sanitize_config NULLS it, so in practice every real
    # export is "unknown". That is reported, not guessed: it is the reason
    # helix writes its own eval artifact (see `save`).
    src = full.get("weight") or (full.get("model") or {}).get("checkpoint")
    which = "unknown"
    if src:
        which = "ema" if "ema" in os.path.basename(str(src)).lower() else "raw"
    tok = export_tokenizer_cfg(path)
    return Artifact(arch=arch, op=_op(tok, n_bands=n_band),
                    weights=which, weights_source=src,
                    provenance={"step": full.get("step"),
                                "tokenizer_extra": _tok_extra(tok)},
                    source=path, fmt="pimm-export", state_dict=sd)


#: Buffers excluded from :func:`weights_digest`: step counters registered as
#: persistent buffers. Including them would make the digest partly a function of
#: how long training ran, so the same weights reached by two paths would hash
#: differently.
_DIGEST_SKIP = ("num_batches_tracked", "n_averaged")


def weights_digest(sd):
    """blake2b over a weight set -- the only identifier that survives a move.

    ``weights_source`` is a path the export recorded and ``weights_are_ema`` is a
    substring test on a filename; neither survives a file being moved, renamed or
    re-exported. This does, and it needs no cooperation from whoever wrote the
    file: if two rows claim the same checkpoint, this is what shows they probed
    the same tensors.

    THERE MUST BE ONE OF THESE. There were briefly two -- a sha256 over the saved
    state_dict in the export script and this blake2b over the loaded model in
    run_probe -- and both landed in the same results row under different names,
    so a reader had two answers to "which weights" and no way to tell which. That
    is the same failure as eight loaders for eight formats, in miniature.

    The dtype is hashed, because two tensors with identical bytes under
    different dtypes are different weights. The retired converter wrote a digest
    of the same shape but a different value (no dtype, no exclusions); rows
    predating this carry those, so compare digests only within a format.

    Bytes go through ``flatten().view(torch.uint8)`` rather than ``.numpy()``:
    ``.numpy()`` raises on bfloat16, and ``view(torch.uint8)`` raises on a 0-dim
    tensor of a different element size, so a scalar buffer would crash the row.
    """
    import hashlib
    import torch

    h = hashlib.blake2b(digest_size=16)
    for k in sorted(sd):
        if k.endswith(_DIGEST_SKIP):
            continue
        v = sd[k]
        if not torch.is_tensor(v):
            continue
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(v.detach().cpu().contiguous().flatten()
                 .view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _jsonable(v, where):
    """Numbers, lists and dicts -- with tensors and arrays spelled out in full.

    Written because ``json.dump(..., default=str)`` silently turned m113's bin
    EDGES into the string repr of a tensor. The artifact wrote, loaded, and
    reported its operating point correctly; it just could not build a model any
    more, because the categorical head's edges are training-set statistics that
    nothing can re-derive. A serialiser that cannot fail is a serialiser that
    loses data, so anything not representable raises here instead.
    """
    import numpy as np

    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x, f"{where}.{k}") for k, x in v.items()}
    if hasattr(v, "detach"):                      # torch tensor
        v = v.detach().cpu().numpy()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, f"{where}[]") for x in v]
    raise TypeError(
        f"{where}: {type(v).__name__} cannot be written to an artifact. Convert "
        f"it to numbers, or leave it out -- an artifact that stringifies what it "
        f"cannot represent is an artifact that has lost it.")


def save(out_dir, *, state_dict, arch, op, weights, provenance=None):
    """Write a helix EVAL artifact: weights, architecture, operating point.

    Deliberately not a training config. This must be scoreable on a machine that
    has never seen this filesystem, so nothing that names a path here is load-
    bearing, and ``weights`` is recorded explicitly because it is the one thing
    ``pimm export`` structurally cannot tell us.
    """
    from safetensors.torch import save_file

    assert weights in ("ema", "raw"), (
        f"weights={weights!r}: an artifact WE write must say which weight set it "
        f"holds. 'unknown' is what we are fixing.")
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state_dict.items()},
              os.path.join(out_dir, _ARTIFACT_WEIGHTS))
    meta = dict(arch=_jsonable(dict(arch), "arch"),
                operating_point=_jsonable(asdict(op), "operating_point"),
                weights=weights,
                # Provenance is free-form and may hold paths, timestamps and a
                # git record, so it is the one place a str() fallback is right:
                # nothing BUILDS from it. Everything above must round-trip.
                provenance=json.loads(json.dumps(dict(provenance or {}),
                                                 default=str, sort_keys=True)))
    tmp = os.path.join(out_dir, _ARTIFACT_JSON + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    os.replace(tmp, os.path.join(out_dir, _ARTIFACT_JSON))
    return out_dir


def build(art, *, device=None, eval_mode=True):
    """Instantiate ``art``'s architecture and, when it carries them, its weights."""
    import torch
    from helix.model import build_fm
    from helix.model.checkpoint import load_state_dict

    # compile_blocks is how the training run executed, not what it learned; an
    # eval rebuild should not pay minutes of compilation for it.
    model = build_fm({k: v for k, v in art.arch.items() if k != "compile_blocks"})
    if art.state_dict is not None:
        load_state_dict(model, art.state_dict, bins=art.op.bins)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def load_bins(path):
    """Bin edges from either a bins sidecar or a checkpoint that carries them.

    The sidecar is its own shape — `scripts/derive_coeff_bins.py` writes a bare
    ``edges`` mapping, which is not a checkpoint and has no architecture — so it
    is read here rather than pushed through :func:`inspect`. Everything else is a
    checkpoint and goes to the one reader, which is this module by design.

    The directory test comes FIRST and is not cosmetic: an eval artifact is a
    directory, and ``torch.load`` on one raises ``IsADirectoryError`` before any
    of the code that would have coped. That is the same shape of bug as
    ``load_probe_model`` learning about export dirs while its sibling did not.

    It lives HERE rather than in ``helix.integrations.pimm.model``, where it was
    written, because it needs nothing from pimm -- only os, torch and this
    module. Importing it from there dragged in
    ``helix/integrations/pimm/__init__.py``, which applies import-time patches to
    ``pimm.engines``; in an environment with no pimm that is a
    ``ModuleNotFoundError``. The image deliberately ships no pimm, so
    ``scripts/smoke_train_fm.py`` -- the last stage of the documented no-data
    smoke path -- could not run there at all.
    """
    import os

    if not os.path.isdir(path):
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "edges" in blob:      # the sidecar
            return blob
    bins = inspect(path).op.bins
    if bins is None:
        raise ValueError(
            f"{path}: no bin edges found. Expected a bins sidecar (an 'edges' "
            f"mapping from scripts/derive_coeff_bins.py), or a checkpoint that "
            f"carries them — an eval artifact does, and a `pimm export` keeps "
            f"them in the weights as the persistent `bin_edges` buffer, so it "
            f"reports none here and needs none.")
    return bins
