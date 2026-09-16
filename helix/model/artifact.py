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
#: The shapes that carry enough to score a number. The rest are refused by name.
READABLE = ("helix-eval", "pimm-export", "converted", "research")


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

    def comparable_to(self, other):
        """Two numbers may be compared only if these agree."""
        return (self.cell_t == other.cell_t and self.pw == other.pw
                and self.pt == other.pt and self.n_bands == other.n_bands)


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
    #: A second weight set, when the source carries one. Only the converted
    #: blobs do: they keep `state_dict_ema` beside the raw weights. An export
    #: directory holds exactly one set, which is why `weights` above is tri-state
    #: rather than a preference.
    state_dict_ema: dict | None = None

    def pick(self, prefer="raw"):
        """``(state_dict, used)``. Falls back to raw LOUDLY, via ``used``.

        A caller that asked for the EMA and silently got the raw weights is
        reporting EMA numbers it did not compute: on a flat-LR WSD run the raw
        weights sit at full LR noise for the whole stable phase, which is what
        the EMA exists to avoid.
        """
        if prefer == "ema" and self.state_dict_ema:
            return self.state_dict_ema, "ema"
        if self.fmt in ("helix-eval", "pimm-export"):
            # One set, described rather than selected. "unknown" propagates: a
            # `pimm export` cannot say what it holds, and calling that "raw"
            # would let an EMA arm and a raw arm be compared in silence.
            return self.state_dict, self.weights
        return self.state_dict, "raw"


def detect(path):
    """Which of :data:`FORMATS` ``path`` is. Never guesses beyond them."""
    path = str(path)
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
    "weights:\n"
    "    pimm export --run-dir <save_path> model_ema.pth <out_dir>\n"
    "then pass <out_dir>.\n"
    "NOT tools/convert_fm_ckpt.py -- that is frozen as the one-time rescue of "
    "the historical m113 checkpoint, which could not describe itself, and it "
    "cannot read a pimm checkpoint at all (it expects 'model'/'ema' keys; pimm "
    "writes 'state_dict').")


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
    if fmt == "pimm-export":
        return _read_pimm_export(path, weights)
    return _read_blob(path, weights, fmt)


def _op(tok, *, n_bands=None, bins=None):
    tok = dict(tok or {})
    n = tok.get("n_bands", n_bands)
    return OperatingPoint(cell_t=tok.get("cell_t"), pw=tok.get("pw"),
                          pt=tok.get("pt"),
                          n_bands=None if n is None else int(n), bins=bins)


def _read_helix_eval(path, weights):
    meta = json.load(open(os.path.join(path, _ARTIFACT_JSON)))
    sd = None
    if weights:
        from safetensors.torch import load_file
        sd = load_file(os.path.join(path, _ARTIFACT_WEIGHTS))
    return Artifact(arch=dict(meta["arch"]),
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
    """
    cfg_path = _export_config_path(path)
    if cfg_path is None:
        return None
    for t in (json.load(open(cfg_path)).get("transform") or []):
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
    return Artifact(arch=arch, op=_op(export_tokenizer_cfg(path), n_bands=n_band),
                    weights=which, weights_source=src,
                    provenance={"step": full.get("step")},
                    source=path, fmt="pimm-export", state_dict=sd)


def _read_blob(path, weights, fmt):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if fmt == "research":
        # The historical shape. It genuinely cannot describe itself — that is
        # what convert_fm_ckpt.py exists for — so say so rather than half-read it.
        raise ValueError(
            f"{path} is a research-tier checkpoint ('model'/'ema' keys). It "
            f"records no tokenizer and no operating point, so a number scored "
            f"from it is unattributable. Convert it once with "
            f"tools/convert_fm_ckpt.py --train-config <the run's yaml>, which is "
            f"kept for exactly this checkpoint.")
    arch = dict(blob.get("config") or {})
    if isinstance(arch.get("film"), list):
        arch["film"] = tuple(arch["film"])
    prov = dict(blob.get("provenance") or {})
    return Artifact(arch=arch,
                    op=_op(blob.get("tokenizer"), n_bands=arch.get("n_band"),
                           bins=blob.get("bins")),
                    weights=str(prov.get("weights", "unknown")),
                    weights_source=prov.get("source"),
                    provenance=prov, source=path, fmt=fmt,
                    state_dict=blob.get("state_dict") if weights else None,
                    state_dict_ema=((blob.get("state_dict_ema") or blob.get("ema"))
                                    if weights else None))


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
    meta = dict(arch={k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in dict(arch).items()},
                operating_point=asdict(op), weights=weights,
                provenance=dict(provenance or {}))
    tmp = os.path.join(out_dir, _ARTIFACT_JSON + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, os.path.join(out_dir, _ARTIFACT_JSON))
    return out_dir


def build(art, *, prefer="raw", device=None, eval_mode=True):
    """Instantiate ``art``'s architecture and, when it carries them, its weights.

    ``prefer`` selects between the two weight sets a converted blob may carry;
    ask ``art.pick(prefer)[1]`` for which one was actually used.
    """
    import torch
    from helix.model import build_fm
    from helix.model.checkpoint import load_state_dict

    model = build_fm(dict(art.arch))
    if art.state_dict is not None:
        load_state_dict(model, art.pick(prefer)[0], bins=art.op.bins)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model
