"""Make helix's coefficient FM resolvable from a pimm config.

Importing this module registers three names into pimm's registries:

    CoeffTokenize     TRANSFORMS  coeff rows -> patch tokens (helix.model.tokenize)
    CoeffTPCDataset   DATASETS    the corpus, read by pimm-data
    Coeff-FM          MODELS      helix.model.build_fm, + converted checkpoint

**pimm is not modified.** A config pulls this in with mmcv's standard
out-of-tree hook, which ``pimm.utils.config.Config.fromfile`` already honours::

    custom_imports = dict(imports=["helix.integrations.pimm"],
                          allow_failed_imports=False)

That keeps the dependency pointing one way — helix knows how to plug into pimm,
pimm knows nothing about helix — so this survives pimm's own churn (its
``migration/warpconvnet`` branch is mid-flight) without a PR into a framework
that serves many detectors, and without adding helix to its CI.

When that migration lands, its registry resolves plain import paths
(``type: "helix.model.fm:FMModel"``), and the MODELS entry below becomes
unnecessary. The TRANSFORMS and DATASETS entries stay, because they adapt rather
than merely name.

Importing this needs pimm AND pimm-data installed; helix's own DSP, tokenizer and
model do not.
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch.optim.lr_scheduler import LambdaLR as _LambdaLR

from pimm.datasets.builder import DATASETS
from pimm.datasets.transform import Compose
from pimm.datasets.transform.common import TRANSFORMS
from pimm.distributed import unwrap_model
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.engines.train import TRAINERS, Trainer
from pimm.models.builder import MODELS
from pimm.utils import comm
from pimm.utils.optimizer import OPTIMIZERS
from pimm.utils.scheduler import SCHEDULERS
from torch.utils.data import Dataset

from helix.integrations._bootstrap import (bootstrap_block, has_bootstrap,
                                           running_roots)
from helix.model.mup import expand_max_lr, param_group_ratios
from helix.model.tokenize import CoeffTokenize

__all__ = ["CoeffTokenize", "CoeffCollect", "CoeffTPCDataset", "CoeffFM",
           "build_coeff_fm",
           "FMTrainer", "CoeffFMEvaluator", "WSDStableLR", "WSDCooldownLR", "WeightEMA",
           "HelixPathBootstrap"]

# ── make pimm's resume survive its own checkpoint loader ─────────────────────
def _patch_rng_restore_to_cpu():
    """Move the saved RNG state back to the CPU before pimm restores it.

    pimm loads a legacy checkpoint with

        map_location = lambda storage, loc: storage.cuda()   (checkpoints.py:940)

    which moves EVERY tensor in the payload to the GPU, including the saved RNG
    state. Both ``torch.set_rng_state`` and ``torch.cuda.set_rng_state_all``
    require CPU ByteTensors, so ``resume=True`` dies with

        TypeError: RNG state must be a torch.ByteTensor

    before the first step -- which is why a preempted run could not pick up where
    it left off, only warm-start from the weights with its optimizer and step
    counter reset. Verified: the payload on disk is a uint8 CPU tensor, the same
    read through pimm's map_location is uint8 on cuda:0, and ``.cpu()`` is
    accepted.

    Patching this ONE name is sufficient and deliberate:
    ``restore_distributed_rng_state`` lives in the same module and resolves
    ``restore_rng_state`` from module globals at CALL time, so the reference
    ``checkpoints.py`` imported at module load picks up the patched version too
    (both checked). Idempotent; only ever moves a tensor that has to be on the
    CPU anyway.
    """
    from pimm.engines import _train_utils as tu

    if getattr(tu.restore_rng_state, "_helix_cpu_shim", False):
        return
    orig = tu.restore_rng_state

    def restore_rng_state(state):
        if state:
            state = dict(state)
            for key in ("torch", "torch_cuda"):
                v = state.get(key)
                if isinstance(v, (list, tuple)):
                    state[key] = [x.cpu() for x in v]
                elif v is not None and getattr(v, "device", None) is not None:
                    state[key] = v.cpu()
        return orig(state)

    restore_rng_state._helix_cpu_shim = True
    tu.restore_rng_state = restore_rng_state


_patch_rng_restore_to_cpu()


# The tokenizer needs no adapter — it is already a duck-typed transform
# (``scope`` + ``__call__(dict) -> dict``), which is why it can live in helix
# with no pimm import. Only the NAME has to reach pimm's registry.
TRANSFORMS.register_module(module=CoeffTokenize, name="CoeffTokenize")


@TRANSFORMS.register_module()
class CoeffCollect:
    """Tokenised part -> the flat tensor dict ``FMModel.forward`` consumes.

    The terminal per-event transform, and it exists for three specific reasons
    that only show up when you run pimm's actual collate over a real sample:

    1. **Flatten.** ``CoeffTokenize`` leaves tokens nested under its part, but
       the model reads ``plane_id``/``inp``/... at the top level.

    2. **Tensorise.** pimm's ``collate_fn`` CONCATENATES tensor leaves
       (``torch.cat``) but sends anything else to ``default_collate``, which
       STACKS. Handing it numpy therefore produced ``inp`` of shape
       ``(1, n_cells, n_slot)`` — a spurious batch dimension the model cannot
       consume. Converting here puts us on the concatenating path, which is also
       the one that stays correct if batching ever arrives.

    3. **Drop ``n_cells``.** It is an int, so collate turns it into
       ``tensor([30976])`` while ``make_mask`` does ``torch.rand(n)``.
       ``FMModel.forward`` derives it from ``plane_id.shape[0]`` anyway, which is
       also correct for a concatenated batch.

    Deliberately does NOT emit ``offset``. pimm's ``run_step`` does
    ``if "offset" in input_dict: input_dict["coord"].shape[0]`` — an offset
    without a ``coord`` raises KeyError *after* the forward. The FM has no
    ``coord`` and, having no event separation, requires ``batch_size=1`` anyway
    (see MULTI_EVENT_BATCHING.md).
    """

    scope = "sample"

    #: int/scalar sample fields that must not reach the model as 0-d tensors
    DROP = ("n_cells",)

    # 'ident' is kept by DEFAULT. CoeffCollect rebuilds `out` from scratch and
    # copies only `keep` from the top level, so with keep=("name",) the source
    # identity pimm-data attaches — (run, source_file, event), the thing that
    # lets a probe reach simulation truth without rebuilding the corpus — was
    # silently dropped before it ever reached a batch. pimm's collate handles it:
    # str lists stay lists, the int becomes a tensor.
    def __init__(self, part="coeff", keys=None, keep=("name", "ident")):
        self.part = part
        self.keys = tuple(keys) if keys else None
        self.keep = tuple(keep)

    def __call__(self, data):
        import numpy as np
        import torch

        sub = data.get(self.part)
        if sub is None:
            raise KeyError(
                f"CoeffCollect: no part {self.part!r} in the sample (have "
                f"{sorted(data)}) — it must run AFTER CoeffTokenize")
        out = {}
        for k, v in sub.items():
            if k.startswith("_") or k in self.DROP:
                continue
            if self.keys is not None and k not in self.keys:
                continue
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(np.ascontiguousarray(v))
        for k in self.keep:                     # carry the event id for seeding/logging
            if k in data:
                out[k] = data[k]
        return out


@DATASETS.register_module()
class CoeffTPCDataset(Dataset):
    """The wavelet-coefficient corpus as a pimm dataset.

    Wraps ``pimm_data.CoeffTPCDataset`` as a pure reader (``transform=None``) and
    runs pimm's own transform pipeline on the raw nested sample — the same shape
    pimm's ``lucid_event_ssl.py`` already uses to consume pimm-data, so a config
    author sees one transform registry.

    Each sample is per-coefficient ROWS, not tokens::

        {'coeff': {'band','plane_gid','wire','tau','value','_meta'}, 'name', 'split'}

    ``_meta`` carries the shard tables the tokenizer needs (``gids``,
    ``n_wires``, ``band_lengths``, ``norm_sigma``), so a DataLoader worker is
    self-sufficient.

    **batch_size must be 1.** The FM has no event separation — attention runs
    over whatever tokens it receives — so a larger batch silently trains a model
    whose tokens attend across unrelated events. See ``MULTI_EVENT_BATCHING.md``.
    """

    def __init__(self, data_root, split="", dataset_name="coeff_tpc",
                 modalities=("coeff", "coeff_clean"), transform=None, loop=1,
                 max_len=-1, strict_lengths=True, event_range=None,
                 exclude_range=None, holdout=None, split_role=None):
        super().__init__()
        try:
            from pimm_data import CoeffTPCDataset as _DS
        except ImportError:                       # older layout / partial install
            from pimm_data.coeff import CoeffTPCDataset as _DS
        # Split parameters must be FORWARDED. This wrapper re-declares the inner
        # dataset's signature, so anything added there is invisible here until
        # it is listed — and configs resolve THIS class, not the inner one.
        #
        # That has now bitten twice: event_range/exclude_range (fixed in
        # df14602) and then holdout/split_role, which failed the first real
        # 2-GPU launch with "unexpected keyword argument 'holdout'". The unit
        # tests construct the inner dataset directly and cannot see it. If a
        # third split parameter appears, add it here in the same commit.
        self._inner = _DS(data_root=data_root, split=split,
                          dataset_name=dataset_name, modalities=tuple(modalities),
                          transform=None, loop=loop, max_len=max_len,
                          strict_lengths=strict_lengths,
                          event_range=event_range, exclude_range=exclude_range,
                          holdout=holdout, split_role=split_role)
        self.transform = Compose(transform)

    def __len__(self):
        return len(self._inner)

    def get_data(self, idx):
        """The raw nested sample, untransformed."""
        return self._inner.get_data(idx)

    def __getitem__(self, idx):
        return self.transform(self.get_data(idx))


@MODELS.register_module("Coeff-FM")
class CoeffFM:
    """Factory registered as ``Coeff-FM``.

    A class rather than a function because pimm's registry requires one:
    ``_register_module`` raises ``TypeError: module must be a class``, and
    ``build_from_cfg`` ends in ``obj_cls(**args)``. ``__new__`` returns the
    ``FMModel`` itself, so a config gets a model rather than a wrapper, and
    nothing downstream has to unwrap it.

    Found only by running it: the registry's type check fires at DECORATION
    time, so a function here fails at import of this module — every test that
    read the source instead of importing it stayed green.
    """

    def __new__(cls, checkpoint=None, weights=True, bins=None, **cfg):
        return build_coeff_fm(checkpoint=checkpoint, weights=weights,
                              bins=bins, **cfg)


def build_coeff_fm(checkpoint=None, weights=True, bins=None, **cfg):
    """Build the coefficient FM, optionally restoring a converted checkpoint.

    ``FMModel.forward(batch) -> dict`` already satisfies pimm's Trainer contract
    (``output_dict["loss"]``), so nothing is wrapped.

    Args:
        checkpoint (str | None): a checkpoint from helix's
            ``tools/convert_fm_ckpt.py`` — self-contained, carrying ``config``,
            ``state_dict`` and (for a categorical head) inlined bin ``edges``.
            Its ``config`` supplies the architecture; ``cfg`` overrides fields.
        weights (bool): restore the weights. ``False`` builds the same
            architecture freshly initialised.
        **cfg: architecture kwargs for ``helix.model.build_fm``.
    """
    from helix.model import build_fm

    blob = None      # NOT `blob = bins = None`: that clobbered the caller's bins
    if checkpoint is not None:
        import torch
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "config" not in blob or "state_dict" not in blob:
            raise ValueError(
                f"{checkpoint} is not a converted checkpoint (no config/state_dict). "
                f"Run helix's tools/convert_fm_ckpt.py on the raw research file — "
                f"it also inlines the categorical bin edges, which the raw "
                f"checkpoint does not carry at all.")
        arch = dict(blob["config"])
        if isinstance(arch.get("film"), list):     # torch round-trip makes it a list
            arch["film"] = tuple(arch["film"])
        arch.update(cfg)
        cfg = arch
        if bins is None:
            bins = blob.get("bins")

    if isinstance(bins, str):
        bins = _load_bins(bins)

    model = build_fm(cfg)
    if blob is not None and weights:
        from helix.model.checkpoint import load_converted
        load_converted(model, blob)
    if getattr(model, "n_bins", 0) > 0:
        if bins is None:
            raise ValueError(
                f"n_bins={model.n_bins} (categorical head) but no bin edges were "
                f"supplied. They are TRAINING-SET STATISTICS, not learned "
                f"parameters, so the model cannot invent them: pass "
                f"bins='/path/to/bins.pt' in the model config, or a `checkpoint` "
                f"whose converted blob carries them inline. Derive fresh edges "
                f"for a new corpus with research tier1_setup_bins.py — the ones "
                f"m113 shipped with came from a different noise model.")
        from helix.model.checkpoint import apply_bins
        apply_bins(model, bins)
    return model


def _load_bins(path):
    """Bin edges from either a bins sidecar or a converted checkpoint."""
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "edges" in blob:                       # tier1_setup_bins.py sidecar
        return blob
    if isinstance(blob.get("bins"), dict):    # converted checkpoint
        return blob["bins"]
    raise ValueError(
        f"{path}: no bin edges found (expected an 'edges' key, or a converted "
        f"checkpoint carrying 'bins')")


@TRAINERS.register_module()
class FMTrainer(Trainer):
    """pimm's Trainer with the two things muP needs, and nothing else.

    Config: ``train = dict(type="FMTrainer")``.

    pimm's own machinery covers everything else — DDP, resume, logging, hooks,
    the loop itself. Only the optimizer and the scheduler cannot be expressed in
    config, for one reason each.
    """

    def build_train_loader(self):
        """Refuse a per-GPU batch other than 1.

        pimm's ``batch_size`` is GLOBAL and ``batch_size_per_gpu`` is derived as
        ``batch_size // world_size``. The FM has NO event separation — attention
        runs over whatever tokens it is handed — so a per-GPU batch above 1
        concatenates unrelated events into one token set and silently trains a
        model whose tokens attend across event boundaries. Nothing downstream
        can see that; the loss simply means something else.

        It is easy to hit by accident, because the correct global value tracks
        the GPU count: ``batch_size = 4`` is right on 4 ranks and wrong on 1.
        See MULTI_EVENT_BATCHING.md.
        """
        per_gpu = self.cfg.batch_size // comm.get_world_size()
        if per_gpu != 1:
            raise ValueError(
                f"batch_size={self.cfg.batch_size} over world_size="
                f"{comm.get_world_size()} gives {per_gpu} events per GPU. The FM "
                f"requires exactly 1: it has no event separation, so a larger "
                f"per-GPU batch trains attention ACROSS unrelated events without "
                f"failing. Set batch_size to the number of ranks.")
        return super().build_train_loader()

    def build_optimizer(self):
        """Take param groups from the MODEL rather than from name matching.

        pimm's ``build_optimizer`` groups parameters by substring matches against
        ``cfg.param_dicts`` keywords. muP cannot be written that way: the hidden
        group's ``lr / m`` and its compensating ``weight_decay * m`` come from
        the model's width multiplier, not from anything in a parameter's name.
        ``FMModel.param_groups`` already computes them, and is verified
        bit-identical to the research implementation.
        """
        model = unwrap_model(self.model)          # may be DDP-wrapped by build_model
        if not hasattr(model, "param_groups"):
            raise TypeError(
                f"{type(model).__name__} has no param_groups(); FMTrainer exists "
                f"to use it. Use pimm's DefaultTrainer for models without muP.")
        if self.cfg.param_dicts:
            raise ValueError(
                "param_dicts is set, but FMTrainer takes its groups from "
                "model.param_groups(). Keyword matching cannot express muP, and "
                "having both would silently pick one — remove param_dicts.")

        cfg = dict(self.cfg.optimizer)
        base_lr = cfg.get("lr")
        groups = model.param_groups(base_lr, weight_decay=cfg.get("weight_decay"))
        cfg["params"] = groups                    # type/betas/etc still honoured
        opt = OPTIMIZERS.build(cfg)
        self._mup_ratios = param_group_ratios(opt.param_groups, base_lr)
        self.logger.info(
            f"muP param groups: " + ", ".join(
                f"[{i}] {sum(p.numel() for p in g['params'])/1e6:.1f}M "
                f"lr x{r:.3f} wd={g.get('weight_decay')}"
                for i, (g, r) in enumerate(zip(opt.param_groups, self._mup_ratios))))
        return opt

    def build_scheduler(self):
        """Give the scheduler a PER-GROUP peak LR so muP survives it.

        ``OneCycleLR`` takes ``max_lr`` as a scalar or a per-group list. A scalar
        assigns every group the same peak, which discards muP silently — the
        hidden group would climb to ``base_lr`` instead of ``base_lr / m``.
        """
        ratios = getattr(self, "_mup_ratios", None)
        if ratios and "max_lr" in self.cfg.scheduler:
            self.cfg.scheduler.max_lr = expand_max_lr(
                self.cfg.scheduler.max_lr, ratios)
        return super().build_scheduler()


@SCHEDULERS.register_module()
class WSDStableLR(_LambdaLR):
    """Warmup-stable phase with an ABSOLUTE warmup step count.

    Exists because ``warmup_rate`` cannot express what m113 did.
    ``Trainer.build_scheduler`` (pimm engines/train.py:733) OVERWRITES
    ``cfg.scheduler.total_steps`` with ``iters_per_epoch * epoch`` before
    building, unconditionally — so a config that sets ``total_steps`` has it
    discarded, and any ``warmup_rate`` is reinterpreted against the
    trainer-derived total.

    That is not a hypothetical. Expressing m113's 4,000 warmup steps as
    ``warmup_rate = 4000/1_010_000`` against a 1,500-step run yields **5.94
    steps** of warmup: the LR reaches 1.1e-3 by step 7 instead of step 4000 —
    571x research's value at that point, 5.3x the integrated LR over the run —
    on a cold 12-block d=512 transformer. That is the classic loss-spike
    configuration, and it silently rescales again with any change to max_len,
    epoch or GPU count.

    So warmup is specified in STEPS here and total_steps is ignored entirely
    (the stable phase is flat, so it needs no horizon — which is the point of
    WSD: "flat, no horizon baked in", mae_ddp.py:164).

    Matches research's ``lr * s / warmup`` ramp exactly.
    """

    def __init__(self, optimizer, warmup=4000, total_steps=None, last_epoch=-1):
        # total_steps is accepted and ignored: the trainer injects it whether or
        # not it is wanted, and silently dropping it beats failing on a kwarg
        # the caller never set.
        def stable(s):
            return min(1.0, s / warmup) if warmup > 0 else 1.0

        super().__init__(optimizer=optimizer, lr_lambda=stable,
                         last_epoch=last_epoch)


@SCHEDULERS.register_module()
class WSDCooldownLR(_LambdaLR):
    """The cooldown half of warmup-stable-decay: ``lr * max(floor, 1 - sqrt(p))``.

    m113 trained the STABLE phase — `lr_mode: const`, described in mae_ddp.py as
    "WSD stable phase: flat, no horizon baked in". That is the point of WSD: the
    stable run commits to no total step count, so it can be extended, and the
    decay is a SEPARATE short run started from a stable-phase checkpoint. A
    checkpoint like m113's at 1,010,000 steps is therefore not an annealed model
    and should not be read as one.

    pimm has no equivalent. ``PolyLR`` is ``(1-p)**power``, which is a different
    curve: at p=0.25 research gives 0.500 and PolyLR(power=0.5) gives 0.866.
    ``1 - sqrt(p)`` drops fast and early, which is what a short cooldown wants.

    The warmup-then-constant STABLE phase needs no new code —
    ``MultiStepWithWarmupLR(milestones=[])`` leaves the decay factor at 1.0
    forever, which is exactly it.

    Being a LambdaLR, this scales each param group's own ``base_lr``, so muP's
    per-group ratios survive without the ``max_lr`` expansion OneCycleLR needs.
    """

    def __init__(self, optimizer, total_steps, warmup=0, floor=1e-3,
                 last_epoch=-1):
        def wsd(s):
            if warmup and s < warmup:
                return s / warmup                      # research: lr * s / warmup
            p = (s - warmup) / max(1, total_steps - warmup)
            return max(floor, 1.0 - math.sqrt(min(p, 1.0)))

        super().__init__(optimizer=optimizer, lr_lambda=wsd, last_epoch=last_epoch)


@HOOKS.register_module()
class HelixPathBootstrap(HookBase):
    """Put the helix `sys.path` bootstrap back into the config pimm dumps.

    Without this a chained run cannot resume, and it fails LATE — after job 1 has
    burned its whole wall-clock allocation.

    The cause is that `Config.dump` serialises the RESOLVED dict, not the source:
    `custom_imports` is a dict and survives, while the `sys.path.append` that made
    `helix.integrations.pimm` importable is a STATEMENT and does not. Job 1 loads
    the config from the repo and is fine; every later job takes train.sh's resume
    branch, which loads `${EXP_DIR}/config.py` (train.sh:226) with
    `PYTHONPATH=${EXP_DIR}/code` (train.sh:253) — pimm's own snapshot, which does
    not contain helix. `custom_imports` then raises a bare ImportError with the
    real ModuleNotFoundError swallowed by `import_modules_from_strings`.

    So the dumped config is rewritten here, once, on rank 0. The root is taken
    from the RUNNING helix rather than a constant, so a resumed job re-enters the
    same checkout job 1 used instead of whatever the default happens to point at
    by then; HELIX_ROOT still overrides. This runs in `before_train`, after
    `_train_utils.py:147` has dumped the config.

    It cannot instead snapshot helix into `${EXP_DIR}/code`: that directory is
    built by `cp -r scripts tools pimm` inside pimm's train.sh, and the point of
    this integration is that pimm is not modified.
    """

    def __init__(self, root=None):
        self.root = root

    def before_train(self):
        import os

        if comm.get_rank() != 0:
            return
        path = os.path.join(self.trainer.cfg.save_path, "config.py")
        try:
            with open(path) as fh:
                src = fh.read()
        except FileNotFoundError:
            # No dump means nothing will be resumed from it either.
            return
        if has_bootstrap(src):
            return
        helix_root, pimm_data_root = running_roots()
        helix_root = self.root or helix_root
        tmp = path + ".helixtmp"
        with open(tmp, "w") as fh:
            fh.write(bootstrap_block(helix_root, pimm_data_root) + "\n" + src)
        os.replace(tmp, path)          # atomic: a torn config.py is unresumable
        self.trainer.logger.info(
            f"HelixPathBootstrap: re-added the sys.path bootstrap to {path} "
            f"(helix={helix_root!r}, pimm_data={pimm_data_root!r}) so chained "
            f"jobs can resume")


@HOOKS.register_module()
class WeightEMA(HookBase):
    """Exponential moving average of the weights, written into the checkpoint.

    m113 trained with ``ema: 0.9999`` (half-life ~6931 steps). Research keeps a
    full-state EMA on rank 0, updates it after every optimizer step, and saves it
    into every checkpoint and snapshot (mae_ddp.py:147-155, 201-205, 233).

    It is not an optional polish step for a WSD run. The stable phase is FLAT by
    design — no annealing — so the raw weights stay at full LR noise for the
    whole run. The EMA is what stands in for the annealed model until a cooldown
    is actually run, which is what downstream probing is meant to read. Without
    it, probes score a checkpoint that is noisier than anything research probed.

    Rank 0 only, matching research: the EMA is an artifact, not part of the
    optimisation, so it needs no synchronisation.
    """

    def __init__(self, decay=0.9999, save_freq=None, key="state_dict_ema"):
        self.decay = float(decay)
        self.save_freq = save_freq
        self.key = key
        self._shadow = None
        self._step = 0
        self._pnames = None

    def _path(self):
        import os
        return os.path.join(self.trainer.cfg.save_path, "model", "model_ema.pth")

    def before_train(self):
        """Reload the shadow on resume, so preemption does not reset the average.

        pimm's checkpoint payload has no slot for hook state, so nothing collects
        `state_dict()` below. Rather than change shared infrastructure for one
        consumer, the hook persists itself: at decay 0.9999 the half-life is
        ~6,931 steps, so an EMA that restarts from the current weights on every
        requeue is meaningless on a preemptable queue.
        """
        import os
        import torch
        if comm.get_rank() != 0 or not getattr(self.trainer.cfg, "resume", False):
            return
        path = self._path()
        if not os.path.exists(path):
            self.trainer.logger.info("WeightEMA: resume with no saved EMA; "
                                     "the average restarts from here")
            return
        # Loaded to CPU, then moved to wherever the MODEL lives. Without the
        # move the first update mixes a CPU shadow with CUDA weights and raises
        # "Expected all tensors to be on the same device" — a fresh run never
        # hits it, because there the shadow is cloned from the live model and is
        # already on-device. Reachable only on resume.
        blob = torch.load(path, map_location="cpu", weights_only=False)
        ref = next(self._model().parameters()).device
        self._shadow = {k: v.float().to(ref) for k, v in blob["state_dict"].items()}
        self._step = int(blob.get("step", 0))
        now = int(getattr(self.trainer, "global_step", 0) or 0)
        self.trainer.logger.info(
            f"WeightEMA: resumed from step {self._step} (trainer at {now})")
        if now and abs(now - self._step) > 1:
            self.trainer.logger.warning(
                f"WeightEMA: saved at step {self._step} but training resumes at "
                f"{now} — the average is missing {abs(now - self._step)} steps")

    def _save(self):
        import os
        import torch
        if self._shadow is None:
            return
        path = self._path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save({"state_dict": self._shadow, "decay": self.decay,
                    "step": self._step}, tmp)
        os.replace(tmp, path)          # atomic: preemption cannot truncate it

    def _model(self):
        return unwrap_model(self.trainer.model)

    def _averaged_names(self):
        """Names the EMA should actually average: PARAMETERS only.

        Buffers are copied verbatim from the live model instead. Averaging one is
        at best meaningless, and here it was actively wrong: ``bin_edges`` is a
        constant table of training-set statistics that rides in the state_dict
        (so a checkpoint carries its own bin scheme — see ``set_bins``).

        ``sh.mul_(d).add_(v, alpha=1-d)`` with ``sh == v`` is the identity in
        exact arithmetic but NOT in float32: ``v*0.9999`` and ``v*1e-4`` each
        round, and for a generic edge value the two do not sum back to ``v``. The
        error is a deterministic drift to a nearby fixed point, not a random walk
        — measured on a real table it reaches 1.9e-4 within 400 steps and settles
        at 9.1e-4, i.e. 0.83% of a 0.11-wide bin, by ~4,000. That is what the
        shipped EMA exports carry while the raw exports are exact, so the EMA and
        raw weights of one run bin against slightly different schemes.

        The +-1e18 open edges do NOT drift (their ulp swamps the increment), and
        edges at exactly-representable values do not either — which is why this
        needs a realistic table to reproduce at all.

        This model has no BatchNorm (``sync_bn=False``, none in the module tree),
        so no running statistic actually WANTS averaging. If one is ever added it
        must be listed here deliberately rather than picked up by accident.
        """
        if self._pnames is None:
            self._pnames = {n for n, _ in self._model().named_parameters()}
        return self._pnames

    def after_step(self):
        if comm.get_rank() != 0:
            return
        self._step = int(getattr(self.trainer, "global_step", self._step + 1))
        sd = self._model().state_dict()
        if self._shadow is None:
            self._shadow = {k: v.detach().clone().float() for k, v in sd.items()}
            return
        d = self.decay
        avg = self._averaged_names()
        for k, v in sd.items():
            sh = self._shadow.get(k)
            if sh is not None and sh.device != v.device:
                sh = sh.to(v.device)                  # belt and braces
                self._shadow[k] = sh
            if sh is None or k not in avg or not v.is_floating_point():
                self._shadow[k] = v.detach().clone().float()
            else:
                sh.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
        self._maybe_save()

    def state_dict(self):
        """Picked up by the checkpoint payload if the trainer collects hooks."""
        return {} if self._shadow is None else \
            {k: v.clone() for k, v in self._shadow.items()}

    def _maybe_save(self):
        if self.save_freq and self._step and self._step % int(self.save_freq) == 0:
            self._save()

    def after_train(self):
        """Write the EMA beside the final checkpoint.

        Written separately rather than relying on the trainer's payload:
        pimm's build_checkpoint_payload has no hook-state slot, so an EMA that
        only lived in memory would evaporate at the end of the run.
        """
        if comm.get_rank() != 0 or self._shadow is None:
            return
        self._save()
        self.trainer.logger.info(
            f"WeightEMA(decay={self.decay}, step={self._step}) -> {self._path()}")


def p_dev(t):
    """``device_type`` string for ``torch.autocast``; it rejects a full device."""
    return t.device.type


def _acc_grid_free(core, B, mask, logits, gf):
    """Accumulate var_expl / charge-closure sums for one eval batch.

    Both are computed on the SAME support the value loss uses
    (masked & valid & truly-occupied), and both are pooled over tokens by the
    caller — per-batch means would reweight events by density.

    Two read-outs, deliberately:
      * ``cent_asinh`` -> the posterior mean in TOKEN space, for var_expl. That
        is the space the target lives in, so the comparison needs no sinh.
      * ``cent_ratio`` -> the posterior mean of ``raw/sigma`` = E[sinh t], for
        charge. Applying sinh to the asinh-space mean instead is Jensen-biased
        ~31% low; see ``tokenize.decode_categorical``.
    """
    # No presence/finiteness test: `set_bins` DERIVES any centroid table it is
    # not given, so both are populated whenever `bin_edges` is. The guard that
    # used to stand here returned silently, and when a caller passed centroids
    # positionally into the wrong slot every charge metric vanished from the log
    # with nothing to say it had. If a table is NaN now that is a real bug and
    # the metrics should come out NaN and say so.
    ca, cr = core.bin_cent_asinh, core.bin_cent_ratio
    tgt, occ_t, valid = B["tgt"], B["occ"].bool(), B["valid"].bool()
    sel = mask[:, None] & valid & occ_t
    if not bool(sel.any()):
        return
    band = B["band_id"].long()
    # `einsum` is on autocast's lower-precision list, so under the trainer's
    # autocast context these contractions run in bf16 (~3e-3 relative) even with
    # float32 inputs — measured on A100. The elementwise form is float32 but
    # materialises a (n_cells, n_slot, K) product, ~17 MB more at eval shapes;
    # disabling autocast keeps einsum's memory AND float32.
    with torch.autocast(device_type=p_dev(logits), enabled=False):
        p = torch.softmax(logits.float(), -1)                 # (n_cells, n_slot, K)
        rec_a = torch.einsum("csk,ck->cs", p, ca[band].float())
        rec_r = torch.einsum("csk,ck->cs", p, cr[band].float())
    y = tgt.float()[sel]
    d = rec_a[sel] - y
    gf["sse"] += float((d * d).sum())
    gf["sy"] += float(y.sum())
    gf["syy"] += float((y * y).sum())
    gf["nv"] += float(sel.sum())
    # UNSIGNED is the primary: the signed sum of a near-symmetric coefficient
    # distribution is a small difference of large numbers, so the signed ratio
    # swings wildly (a perfectly-binned predictor scored 0.008 on it in test).
    # Research reports both for exactly this reason; the signed one is kept as a
    # bias indicator, not as a closure measure.
    r_sel, t_sel = rec_r[sel], torch.sinh(y)
    gf["chg_pred"] += float(r_sel.abs().sum())
    gf["chg_true"] += float(t_sel.abs().sum())
    gf["chg_pred_s"] += float(r_sel.sum())
    gf["chg_true_s"] += float(t_sel.sum())


@HOOKS.register_module()
class CoeffFMEvaluator(HookBase):
    """Validation for the coefficient FM.

    Deliberately not pimm's ``MAEEvaluator``: that one passes ``return_pred=``
    (which ``FMModel.forward`` does not take) and reports ``coord_loss`` /
    ``feat_loss``, which are a point-cloud MAE's quantities. Ours are occupancy
    BCE and coefficient value — renaming them to match would make the numbers
    misleading rather than compatible.

    Masks are drawn from a SEEDED generator keyed on the batch index, so the
    same events are masked the same way at every evaluation. Without that the
    metric moves epoch to epoch because the mask moved, which reads as model
    variance. (Research's eval intended this and did not get it: its 'plane' mode
    ignored the generator and drew from the global RNG — see helix.model.mask.)

    Publishes ``-avg_loss`` as ``neg_val_loss`` so pimm's CheckpointSaver can
    select on it (higher is better, by its convention).
    """

    def __init__(self, every_n_steps=0, max_batches=None, mask_seed=7,
                 grid_free=True):
        self.every_n_steps = int(every_n_steps)
        self.max_batches = max_batches
        self.mask_seed = int(mask_seed)
        # var_expl + charge closure, computed from the SAME forward as the
        # loss (see _forward) — so this is free, not a second pass. Silently
        # inert unless the checkpoint carries the bin centroids.
        self.grid_free = bool(grid_free)

    def after_step(self):
        if self.every_n_steps <= 0:
            return
        step = (self.trainer.comm_info["iter"]
                + self.trainer.comm_info["iter_per_epoch"] * self.trainer.comm_info["epoch"])
        if (step + 1) % self.every_n_steps == 0:
            self.eval()

    def after_epoch(self):
        if self.every_n_steps <= 0:
            self.eval()

    def eval(self):
        """Rank 0 evaluates; the others wait at a barrier.

        The barrier is symmetric or it deadlocks. Every non-zero rank blocks in
        ``comm.synchronize()``, so rank 0 MUST rejoin on every exit path — the
        try/finally, not a call at the end of the happy path. Getting this wrong
        does not fail: the run completes training and evaluation, logs a final
        checkpoint, and then hangs forever, because rank 0 walks into the
        collective checkpoint save while the others are still in the barrier.
        That is exactly what happened on the first 2-GPU run.
        """
        world = comm.get_world_size()
        if comm.get_rank() != 0:
            if world > 1:
                comm.synchronize()
            return
        try:
            self._eval_rank0()
        finally:
            if world > 1:
                comm.synchronize()          # rejoin, whatever happened above

    def _forward(self, model, core, B, mask, gf):
        """One forward per batch: the loss AND the grid-free metrics from the
        same heads.

        The obvious version calls ``model(B, tok_mask=mask)`` for the loss and
        ``core.raw_heads(B, mask)`` again for the metrics — a SECOND full
        encoder+decoder pass. Evaluation is not cheap here: measured on
        `coeff-fm-train`, eval is 43.8 s x 100 evals = 73 min of a 469 min run,
        i.e. **15.5% of wall time**. Doubling it would cost ~15% of every future
        run to compute two scalars.

        So for the categorical head — the only one we train — reproduce
        ``FMModel.forward``'s branch here from a single ``raw_heads`` call. Any
        other head configuration falls back to ``forward``, which keeps this
        correct for heads it does not know about rather than silently scoring
        them with the wrong loss.

        Calling ``core`` rather than ``model`` skips the DDP wrapper; under
        ``no_grad`` in eval there is no gradient to synchronise, and the previous
        code already reached for ``core.raw_heads`` for the same reason.
        """
        if self.grid_free and getattr(core, "n_bins", 0) > 0:
            from helix.model.loss import losses_cat
            assert torch.isfinite(core.bin_edges).all(), \
                ("n_bins > 0 requires set_bins(edges) before evaluation — the "
                 "edges buffer is still unset (NaN).")
            # The same check `forward` runs. Skipping `forward` to save a second
            # pass must not also skip its contract.
            core.require_batch_keys(B)
            occ, val, _ = core.raw_heads(B, mask)
            bce, vloss = losses_cat(occ, val, B, mask, core.bin_edges,
                                    vis_w=core.vis_w)
            _acc_grid_free(core, B, mask, val, gf)
            return {"loss": bce + vloss, "bce": bce.detach(),
                    "val": vloss.detach(),
                    "masked_frac": mask.float().mean().detach()}
        return model(B, tok_mask=mask)

    def _eval_rank0(self):
        loader = getattr(self.trainer, "val_loader", None)
        if loader is None:
            self.trainer.logger.info("CoeffFMEvaluator: no val_loader; skipping")
            return

        from pimm.distributed import move_batch_to_device
        self.trainer.logger.info(">>>>>>>> Coeff FM validation >>>>>>>>")
        model = self.trainer.model
        was_training = model.training
        model.eval()
        core = unwrap_model(model)
        device = self.trainer.parallel_context.device

        totals, counts, n = {}, {}, 0
        gf = {k: 0.0 for k in ("sse", "sy", "syy", "nv", "chg_pred",
                               "chg_true", "chg_pred_s", "chg_true_s")}
        with torch.no_grad():
            for i, input_dict in enumerate(loader):
                if self.max_batches is not None and i >= self.max_batches:
                    break
                B = move_batch_to_device(input_dict, device)
                B.setdefault("n_cells", B["plane_id"].shape[0])
                gen = torch.Generator(device=device).manual_seed(self.mask_seed + i)
                # mode="random" EXPLICITLY, matching research: mae_ddp.py:216
                # passes args.mask_mode to perband_mse_cat, so the eval metric is
                # a pure random-mask number even when plane_frac > 0 (research
                # reports the plane-masked number as a SEPARATE curve, and only
                # for non-categorical heads — m113 is categorical, so its eval
                # was pure random). Without this, plane_frac makes val loss a
                # 90/10 mixture and it stops being comparable across runs.
                mask = core.make_mask(B, mode="random", gen=gen)   # SAME every eval
                ctx = (torch.autocast(
                           device_type=device.type,
                           dtype=(torch.bfloat16
                                  if self.trainer.cfg.amp_dtype == "bfloat16"
                                  else torch.float16))
                       if getattr(self.trainer.cfg, "enable_amp", False)
                       else contextlib.nullcontext())
                with ctx:
                    out = self._forward(model, core, B, mask, gf)
                # POOL over tokens, do not average per-event means. Each loss
                # term already divided by ITS OWN support inside the loss, so
                # summing those means weights a 5k-token event equally with a
                # 40k-token one. Dense events are both heavier and harder —
                # corr(per-event CE, token count) = 0.79 — so the unweighted
                # mean is biased optimistic by ~0.09 nats, which is larger than
                # the k30-vs-R1 difference these numbers were used to compare.
                # Research pools (fm/train.py:143, fm/cross_nll.py:41).
                #
                # The supports are recomputed here rather than returned from the
                # loss, so `losses*` stay byte-comparable with research.
                mrow = mask[:, None]
                w_occ = float((mrow & B["valid"].bool()).sum())
                w_val = float((B["occ"].bool() & B["valid"].bool() & mrow).sum())
                wt = {"bce": w_occ, "val": w_val, "masked_frac": float(mask.numel())}
                for k, v in out.items():
                    if torch.is_tensor(v) and v.ndim == 0 and k != "loss":
                        w = wt.get(k, 1.0)
                        totals[k] = totals.get(k, 0.0) + float(v) * w
                        counts[k] = counts.get(k, 0.0) + w
                n += 1

        if was_training:
            model.train()
        if not n:
            self.trainer.logger.info("CoeffFMEvaluator: val_loader was empty")
            return

        # `loss` is rebuilt from the pooled parts, not pooled itself: it is
        # bce + val, and the two have different denominators.
        avg = {k: v / max(counts.get(k, n), 1e-9) for k, v in totals.items()}
        avg["loss"] = avg.get("bce", 0.0) + avg.get("val", 0.0)
        if gf["nv"] > 0:
            # var_expl is GRID-FREE: it compares the posterior-mean asinh
            # reconstruction against the target's own measured variance, so it
            # survives a change of bin table or corpus in a way cross-entropy
            # does not. charge_closure is Sum(pred)/Sum(true) in units of sigma,
            # nominal 1.0. Research computes both every eval
            # (fm/train.py:perband_mse_cat, mae_ddp.py:216-220); their absence
            # here is why a 31%-low charge read-back and a cross-table CE
            # comparison both went unnoticed.
            mean_y = gf["sy"] / gf["nv"]
            var_y = max(gf["syy"] / gf["nv"] - mean_y * mean_y, 1e-12)
            avg["var_expl"] = 1.0 - (gf["sse"] / gf["nv"]) / var_y
            if gf["chg_true"] > 1e-9:
                avg["charge_closure"] = gf["chg_pred"] / gf["chg_true"]
            if abs(gf["chg_true_s"]) > 1e-6 * max(gf["chg_true"], 1e-9):
                avg["charge_bias"] = gf["chg_pred_s"] / gf["chg_true_s"]
        self.trainer.logger.info(
            f"   [coeff-eval] batches={n} " +
            " ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items())))
        writer = getattr(self.trainer, "writer", None)
        if writer is not None:
            # GLOBAL STEP, not epoch. EVAL_EVERY (1178) does not divide
            # iters_per_epoch (4758), so ~4 evals land inside each epoch and
            # three of every four were overwritten — 100 evals collapsed to 25
            # TB points. train.log kept all 100, which is why the curve script
            # parses the log instead. Research keys fm_curve.jsonl on step.
            step = int(getattr(self.trainer, "global_step", 0) or
                       self.trainer.comm_info.get("epoch", 0))
            for k, v in avg.items():
                writer.add_scalar(f"val/{k}", v, step)
        self.trainer.comm_info["current_metric_value"] = -avg["loss"]   # higher is better
        self.trainer.comm_info["current_metric_name"] = "neg_val_loss"
