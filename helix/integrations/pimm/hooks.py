"""Run-lifecycle hooks: the resume bootstrap (+ provenance) and the weight EMA.

Not evaluation — that is :mod:`helix.integrations.pimm.eval`. These two are
about what a run WRITES, the evaluator about what it MEASURES.
"""

from __future__ import annotations

import torch

from pimm.distributed import unwrap_model
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.utils import comm

from helix.integrations._bootstrap import (bootstrap_block, has_bootstrap,
                                           running_roots)


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
        self._stamp_provenance()
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

    def _stamp_provenance(self):
        """Write ``<save_path>/provenance.json``: which code produced this run.

        This hook already resolves the checkouts for the resume bootstrap, so it
        is the one place that knows them — recording them here costs one file and
        no new hook in the config.

        Nothing recorded this before. A run directory held weights, a config and
        a log, and no statement of which helix produced them; reconstructing that
        for m113 meant trying configurations until the goldens matched. Appends
        rather than overwrites, so a chained/requeued job leaves a record of
        every link — a run that was preempted and resumed from a DIFFERENT
        working tree is exactly the case worth catching, and an overwrite would
        hide it.
        """
        import json
        import os

        from helix.integrations._bootstrap import provenance

        try:
            info = provenance()
            info["step"] = int(getattr(self.trainer, "global_step", 0) or 0)
            path = os.path.join(self.trainer.cfg.save_path, "provenance.json")
            log = []
            if os.path.exists(path):
                with open(path) as fh:
                    log = json.load(fh)
                if isinstance(log, dict):          # a single record from an older run
                    log = [log]
            log.append(info)
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(log, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
            h = info["helix"]
            self.trainer.logger.info(
                f"HelixPathBootstrap: provenance -> {path} "
                f"(helix {(h['commit'] or '?')[:12]}"
                f"{'-dirty' if h['dirty'] else ''} on {h['branch']})")
            if h["dirty"]:
                self.trainer.logger.warning(
                    "helix working tree is DIRTY — the commit recorded in "
                    "provenance.json does not fully describe the code that ran")
        except Exception:
            # Provenance is a record, not a dependency. Never take down a run.
            self.trainer.logger.exception(
                "HelixPathBootstrap: could not write provenance.json")


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
