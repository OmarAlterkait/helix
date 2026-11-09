"""Reading a pimm run: the model it trained, the tokenizer it trained with, and what it was.

A run is pimm's ``save_path``. Everything needed to score it is already there, written by pimm
and by helix's hooks, so nothing is exported or promoted first:

    resolved_config.json     pimm's dump of the config the run trained with: the model section
                             (the architecture, ``build_fm``), the train split's CoeffTokenize cfg
                             (the tokenizer), the corpus (data.train)
    model/model_ema.pth      helix.integrations.pimm.hooks.WeightEMA: {state_dict, decay, step}
    model/last/weights.pth   pimm's split checkpoint: {state_dict}; its step is in
                             model/last/trainer.dcp (``trainer.global_step``)
    model/iter_N.pth,        pimm: {state_dict}
    model/model_best.pth     (best: CheckpointSaver's lowest validation loss, when it selects one)
    provenance.json          HelixPathBootstrap: one record per link (code, corpus identity, step)

``load_run(path, weights="ema")`` takes the run directory (``weights`` picks the EMA, the last or
the best checkpoint) or one of those weight files / the ``last`` directory, and returns a
:class:`Run`: the model in eval mode with the bins it trained with (``bin_edges`` and the
centroid tables are persistent buffers, so they ride in every weight file), the tokenizer's
``PatchConfig``, and the run's provenance.

Two kinds of run, by the model section's ``type``: an FM (``Coeff-FM``, ``configs/pimm/fm.py``)
and a denoiser (``Coeff-Denoise``, ``configs/pimm/denoise.py``), whose model names the FM run it
is built on. A denoiser's architecture and tokenizer are that run's, so reading a denoiser reads
the FM run's ``resolved_config.json`` too (its weights are the denoiser's own).

Starting a TRAINING run from a checkpoint is not this module's job: pimm's ``weight=`` does it
(``configs/pimm/fm_cooldown.py``); a denoiser's model initialises its encoder from its FM run
through this loader (:func:`helix.model.denoise.build_denoise`).

No pimm import: pimm is needed to write a run, not to read one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

WEIGHTS = ("ema", "last", "best")
_WEIGHT_PATH = {"ema": "model_ema.pth", "last": "last", "best": "model_best.pth"}
_CONFIG = "resolved_config.json"


@dataclass
class Run:
    """A trained model and what it was trained with."""
    model: object                   # FMModel or DenoiseModel, eval mode, requires_grad off
    tokenizer: object               # PatchConfig
    config: dict                    # resolved_config.json
    provenance: dict = field(default_factory=dict)


def run_dir(path):
    """The run directory (``save_path``) of ``path``: itself or the nearest ancestor holding
    ``resolved_config.json``."""
    p = os.path.abspath(str(path))
    if not os.path.exists(p):
        raise FileNotFoundError(f"no run or checkpoint at {p}")
    d = p if os.path.isdir(p) else os.path.dirname(p)
    for _ in range(4):
        if os.path.isfile(os.path.join(d, _CONFIG)):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError(f"{p}: no {_CONFIG} in it or above it -- not inside a pimm run directory")


#: Training options a run's config may carry that ``build_fm`` no longer accepts: partial
#: decoding (``dec_frac``) and several masks per step (``n_masks``), both training-only.
RETIRED_TRAIN_OPTS = ("dec_frac", "n_masks")


def model_config(config):
    """``build_fm`` kwargs from a resolved config: the model section without the registry
    selector and the bins table (the bins ride in the weights), and without compilation (how the
    run executed, not what it learned). A run may record a training option since removed
    (:data:`RETIRED_TRAIN_OPTS`); it never changed the model outside training, so it is dropped."""
    arch = {k: v for k, v in dict(config["model"]).items()
            if k not in ("type", "bins") + RETIRED_TRAIN_OPTS}
    arch["compile_blocks"] = False
    for k, v in arch.items():
        if isinstance(v, list):                      # JSON has no tuple
            arch[k] = tuple(v)
    return arch


def tokenizer_config(config):
    """The ``PatchConfig`` of the train split's ``CoeffTokenize`` -- what the weights saw. A
    ``cfg="model"`` (a denoiser's) is the tokenizer of the FM run its model is built on
    (``helix.integrations.pimm.trainer.resolve_tokenizer`` resolved it the same way)."""
    from helix.model.tokenize import patch_config
    for t in (config.get("data") or {}).get("train", {}).get("transform") or []:
        if isinstance(t, dict) and t.get("type") == "CoeffTokenize":
            if t.get("cfg") == "model":
                return tokenizer_config(_read_config(config["model"]["fm"]))
            return patch_config(t.get("cfg"))
    raise ValueError("the run's config has no CoeffTokenize in data.train.transform")


def build_model(config):
    """A freshly initialised model of the run's architecture: an FM (no bins), or a denoiser on
    its FM run's architecture (that run's weights NOT loaded)."""
    m = dict(config["model"])
    if m.get("type") == "Coeff-Denoise":
        from helix.model.denoise import build_denoise
        kw = {k: v for k, v in m.items() if k not in ("type", "fm")}
        return build_denoise(m["fm"], **dict(kw, pretrained=False, compile_blocks=False))
    from helix.model import build_fm
    return build_fm(model_config(config))


def _read_config(path):
    return json.load(open(os.path.join(run_dir(path), _CONFIG)))


def load_run(path, weights="ema", *, device=None, fm=None):
    """A :class:`Run` from a run directory, or from one weight file (or ``last`` directory) of it.

    ``weights``: ``"ema"`` (``model/model_ema.pth``), ``"last"`` (``model/last``) or ``"best"``
    (``model/model_best.pth``), used when ``path`` is the run directory; a weight file names
    itself. A missing EMA is an error, not a fallback to the raw weights. ``fm``: for a denoiser,
    where its FM run is at this site when the recorded ``model.fm`` (an absolute path) has moved
    -- its architecture and tokenizer are read from there.
    """
    import torch

    rd = run_dir(path)
    config = json.load(open(os.path.join(rd, _CONFIG)))
    if fm is not None:
        config["model"]["fm"] = str(fm)
    p = os.path.abspath(str(path))
    if p == rd:
        if weights not in WEIGHTS:
            raise ValueError(f"weights={weights!r}: one of {WEIGHTS}")
        p = os.path.join(rd, "model", _WEIGHT_PATH[weights])
        if not os.path.exists(p):
            raise FileNotFoundError(f"{rd}: no {os.path.relpath(p, rd)} (weights={weights!r})")
    name = os.path.basename(p)
    kind = {v: k for k, v in _WEIGHT_PATH.items()}.get(name, name)
    wfile = os.path.join(p, "weights.pth") if os.path.isdir(p) else p
    blob = torch.load(wfile, map_location="cpu", weights_only=False)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    step = blob.get("step") if isinstance(blob, dict) else None
    if step is None and kind == "last":
        step = _dcp_step(os.path.join(p, "trainer.dcp"))
    if step is None:
        m = re.fullmatch(r"iter_(\d+)\.pth", os.path.basename(wfile))
        step = int(m.group(1)) if m else None

    model = build_model(config)
    model.load_state_dict(sd, strict=True)
    if hasattr(model, "bin_edges") and not torch.isfinite(model.bin_edges).all():
        raise ValueError(f"{wfile}: its bin_edges are unset -- these weights cannot be scored")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    links = []
    pp = os.path.join(rd, "provenance.json")
    if os.path.exists(pp):
        links = json.load(open(pp))
    last = links[-1] if links else {}
    train = (config.get("data") or {}).get("train", {})
    prov = dict(run=rd, weights=kind, checkpoint=wfile, step=None if step is None else int(step),
                bins=config["model"].get("bins"),
                corpus=dict(data_root=train.get("data_root"), split=train.get("split"),
                            dataset_name=train.get("dataset_name"), holdout=train.get("holdout"),
                            identity=last.get("corpus")),
                code={k: last.get(k) for k in ("helix", "pimm", "pimm_data")}, links=len(links))
    return Run(model=_frozen(model, dev), tokenizer=tokenizer_config(config), config=config, provenance=prov)


def _frozen(model, device):
    model.to(device).eval()
    for q in model.parameters():
        q.requires_grad_(False)
    return model


def _dcp_step(trainer_dir):
    """``trainer.global_step`` from a pimm split checkpoint's trainer-state DCP, or None."""
    if not os.path.isdir(trainer_dir):
        return None
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    state = {"trainer": {"global_step": 0}}
    dcp.load(state, checkpoint_id=trainer_dir, planner=DefaultLoadPlanner(allow_partial_load=True))
    return int(state["trainer"]["global_step"])


def apply_bins(model, bins, *, log=None):
    """Install a bin table onto ``model``. The ONLY caller of ``set_bins``.

    ``bins`` is the mapping ``helix.data.bins`` writes (``helix.data.bins.load``): ``edges``
    required, ``cent_asinh``/``cent_ratio`` optional (``set_bins`` derives what is absent); any
    other key is ignored. When the model already holds finite edges, the largest disagreement is
    reported before overwriting: replacing a checkpoint's own edges with another table's is how a
    model gets evaluated against a grid it never saw.
    """
    import torch

    assert "edges" in bins, f"bin table has no 'edges' (keys: {sorted(bins)})"
    prev = getattr(model, "bin_edges", None)
    if prev is not None and torch.isfinite(prev).all():
        d = (prev.detach().cpu() - torch.as_tensor(bins["edges"], dtype=prev.dtype)).abs().max()
        msg = f"apply_bins: overwriting existing bin_edges, max|delta| = {d:.3e}"
        if d > 0 and log is not None:
            log(msg)
        elif d > 0:
            import warnings
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return model.set_bins(bins["edges"], cent_asinh=bins.get("cent_asinh"), cent_ratio=bins.get("cent_ratio"))
