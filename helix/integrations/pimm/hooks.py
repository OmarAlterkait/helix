"""Run-lifecycle hooks: provenance (+ the corpus-mismatch refusal) and the weight EMA.
"""

from __future__ import annotations

import math
import torch

from pimm.distributed import unwrap_model
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.utils import comm


@HOOKS.register_module()
class HelixProvenance(HookBase):
    """Record which code and corpus produced this run; refuse to continue it on another corpus.

    Rank 0, in ``before_train`` -- on every link of a chain, so a link resumed from a different
    tree or against a different corpus is on record (and the latter refused).
    """

    def before_train(self):
        if comm.get_rank() != 0:
            return
        self._stamp_provenance()

    def _corpus_identity(self):
        """The DSP identity of the corpus this run is training on, or None.

        The code provenance beside it answers "what produced this run"; without
        this, nothing answers "on WHAT DATA". Two corpora differing only in the
        coherent-removal gate read identically and score differently, so the
        digest is what lets an evaluator refuse a mismatch instead of returning a
        plausible number -- see helix.data.verify.

        Best-effort by design: a run must not die because provenance could not be
        read; an unstamped run is simply unchecked.
        """
        try:
            from helix.data.verify import corpus_identity
            d = (self.trainer.cfg.data or {}).get("train", {})
            root, name = d.get("data_root"), d.get("dataset_name", "sim_wire")
            if not root:
                return None
            split = d.get("split")
            if isinstance(split, (list, tuple)):
                split = split[0] if split else None
            return corpus_identity(root, dataset_name=name, split=split)
        except Exception as e:                      # never fail a run over this
            return {"error": f"{type(e).__name__}: {e}"}

    def _stamp_provenance(self):
        """Write ``<save_path>/provenance.json``: which code produced this run.

        Appends rather than overwrites, so a
        chained/requeued job leaves a record of every link — a run that was
        preempted and resumed from a DIFFERENT working tree is exactly the case
        worth catching, and an overwrite would hide it.
        """
        import json
        import os

        from helix.integrations._bootstrap import provenance

        _deferred_check = None
        try:
            info = provenance()
            info["step"] = int(getattr(self.trainer, "global_step", 0) or 0)
            info["corpus"] = self._corpus_identity()
            path = os.path.join(self.trainer.cfg.save_path, "provenance.json")
            log = []
            if os.path.exists(path):
                with open(path) as fh:
                    log = json.load(fh)
            # REFUSE to continue a run directory against a different corpus.
            # Recording the corpus is not enough: without the check, the training
            # path could resume a checkpoint trained on one corpus against
            # another and report plausible numbers the whole way.
            #
            # The failure is realistic: save_path is derived from a config whose
            # run name is a literal, so pointing HELIX_EXP somewhere new while
            # leaving the name alone lands on an existing run's directory. A
            # corpus swap under a resumed run is the silent version of the very
            # mismatch the identity guard (helix.data.verify) was written for.
            prev = next((r.get("corpus") for r in reversed(log)
                         if isinstance(r, dict) and r.get("corpus")), None)
            if prev and info.get("corpus") and not info["corpus"].get("error"):
                # DEFERRED PAST THE except BELOW, deliberately. That handler
                # ("never take down a run") is right for writing a JSON file and
                # exactly wrong for a corpus refusal, whose entire job is to take
                # down a run: inside the try, the ValueError would be logged and
                # training would continue against the wrong corpus.
                _deferred_check = (prev, info["corpus"], path)

            log.append(info)
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(log, fh, indent=2, sort_keys=True)
            os.replace(tmp, path)
            h = info["helix"]
            self.trainer.logger.info(
                f"HelixProvenance: provenance -> {path} "
                f"(helix {(h['commit'] or '?')[:12]}"
                f"{'-dirty' if h['dirty'] else ''} on {h['branch']})")
            if h["dirty"]:
                self.trainer.logger.warning(
                    "helix working tree is DIRTY — the commit recorded in "
                    "provenance.json does not fully describe the code that ran")
        except Exception:
            # Provenance is a record, not a dependency. Never take down a run.
            # NOTE: the corpus check is deliberately NOT in here -- see below.
            self.trainer.logger.exception(
                "HelixProvenance: could not write provenance.json")

        # OUTSIDE the handler: a corpus mismatch must reach the caller.
        if _deferred_check is not None:
            from helix.data.verify import check_corpus_matches
            prev, actual, path = _deferred_check
            # Raises on a present-and-different basis_digest; a missing or
            # unreadable record is tolerated, exactly as at eval time.
            note = check_corpus_matches(prev, actual, where=path)
            self.trainer.logger.info(f"[corpus] {note}")


@HOOKS.register_module()
class HelixPathBootstrap(HelixProvenance):
    """The old name of :class:`HelixProvenance`, kept so a config dumped by a run started before
    the rename still resolves. It no longer rewrites the dumped config: the launcher exports
    helix on PYTHONPATH for every link (``configs/launch/_common.yaml``)."""

    def __init__(self, root=None):                  # ``root`` was the rewrite's; ignored
        pass


@HOOKS.register_module()
class WeightEMA(HookBase):
    """Exponential moving average of the weights, written beside the checkpoint.

    Decay 0.9999 (half-life ~6931 steps), updated on rank 0 after every optimizer
    step.

    It is not an optional polish step for a WSD run. The stable phase is FLAT by
    design — no annealing — so the raw weights stay at full LR noise for the
    whole run. The EMA is what stands in for the annealed model until a cooldown
    is actually run, which is what downstream probing is meant to read. Without
    it, probes score the raw, LR-noisy weights.

    Rank 0 only: the EMA is an artifact, not part of the optimisation, so it
    needs no synchronisation.
    """

    def __init__(self, decay=0.9999, save_freq=None,
                 max_drift=None, on_drift="discard"):
        self.decay = float(decay)
        self.save_freq = save_freq
        self.max_drift = max_drift
        self.on_drift = on_drift
        self._shadow = None
        self._plan = None     # (shadows, live params, other keys, live tensors), built once
        self._step = 0
        self._pnames = None

    def _path(self):
        import os
        return os.path.join(self.trainer.cfg.save_path, "model", "model_ema.pth")

    def before_train(self):
        """Reload the shadow on resume, so preemption does not reset the average.

        pimm's checkpoint payload has no slot for hook state -- nothing upstream
        ever asks a hook to serialise itself. Rather than change shared
        infrastructure for one consumer, the hook persists itself through
        `_save`/`_load`: at decay 0.9999 the half-life is
        ~6,931 steps, so an EMA that restarts from the current weights on every
        requeue is meaningless on a preemptable queue.
        """
        import os
        import torch
        if self.save_freq == "auto":
            raise RuntimeError("WeightEMA: save_freq='auto' was never resolved; "
                               "it needs train=dict(type='FMTrainer')")
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
        saved_step = int(blob["step"])
        saved_decay = float(blob["decay"])
        now = int(getattr(self.trainer, "global_step", 0) or 0)
        drift = abs(now - saved_step) if now else 0

        # A shadow cannot change its time constant retroactively, so a sidecar
        # written at a different decay is not adopted.
        if saved_decay != self.decay:
            self.trainer.logger.warning(
                f"WeightEMA: sidecar written with decay={saved_decay}, hook "
                f"configured for {self.decay} — discarding; the average restarts.")
            return

        # An INTACT pair drifts by EXACTLY 0. WeightEMA precedes CheckpointSaver
        # in cfg.hooks and both key off trainer.global_step, already advanced by
        # _record_step_state (pimm train.py:400-402); the saver seeds step_count
        # from the same value (pimm hooks/checkpoint.py:83-87) and increments it
        # first. So any non-zero drift is a torn pair.
        #
        # Non-zero drift IS reachable: preemption between the two writes (drift =
        # save_freq), and pimm checkpoints.py:1104-1111, which rewinds global_step
        # to start_epoch*iter_per_epoch when the dataloader cursor is dropped
        # (world_size or num_worker change, or a cursor-less mid-epoch save) —
        # drift up to a full epoch, whose steps are then replayed and averaged
        # in twice.
        #
        # Whether to keep a drifted shadow is set by the decay, not by taste:
        # after K bad steps 1 - decay**K of the mass sits on wrong updates, while
        # DISCARDING costs 100% and needs a half-life to recover. So using beats
        # discarding right up to the half-life, ln(0.5)/ln(decay) = 6,931 steps at
        # 0.9999. Raising instead would kill a multi-day run over an artifact this
        # class's own docstring calls "not part of the optimisation".
        half_life = (math.log(0.5) / math.log(self.decay)
                     if 0.0 < self.decay < 1.0 else float("inf"))
        limit = half_life if self.max_drift is None else float(self.max_drift)
        if drift > limit:
            msg = (f"WeightEMA: sidecar at step {saved_step}, training resumes at "
                   f"{now} (drift {drift} > {limit:.0f}); "
                   f"{1 - self.decay ** drift:.1%} of the average would be "
                   f"duplicated or missing updates. DISCARDING — the average "
                   f"restarts from here.")
            if self.on_drift == "error":
                raise RuntimeError(msg)
            self.trainer.logger.warning(msg)
            return                      # _shadow stays None -> fresh clone

        ref = next(self._model().parameters()).device
        self._shadow = {k: v.float().to(ref) for k, v in blob["state_dict"].items()}
        self._step = saved_step
        if drift:
            self.trainer.logger.warning(
                f"WeightEMA: drift {drift} steps on resume (sidecar {saved_step}, "
                f"trainer {now}); using it — {1 - self.decay ** drift:.2%} of the "
                f"average is duplicated or missing, within the "
                f"{limit:.0f}-step budget")
        else:
            self.trainer.logger.info(
                f"WeightEMA: resumed intact at step {saved_step}")

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
        at 9.1e-4, i.e. 0.83% of a 0.11-wide bin, by ~4,000. The EMA weights would
        carry that while the raw weights are exact, so the EMA and raw weights of
        one run would bin against slightly different schemes.

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
        if self._shadow is None:
            self._shadow = {k: v.detach().clone().float()
                            for k, v in self._model().state_dict().items()}
            return
        d = self.decay
        if self._plan is None:
            # Built ONCE. state_dict() tensors alias the live parameters' storage
            # and the optimizer updates in place, so the references stay current.
            # Rebuilding the dict and looping per tensor every step is hundreds
            # of kernel launches and a Python loop on rank 0 alone -- which, in a
            # launch-bound step, makes rank 0 the straggler every other rank waits
            # for at the all-reduce.
            sd = self._model().state_dict()
            avg = self._averaged_names()
            shadows, live, other = [], [], {}
            for k, v in sd.items():
                sh = self._shadow.get(k)
                if sh is not None and sh.device != v.device:
                    sh = sh.to(v.device)                  # belt and braces
                    self._shadow[k] = sh
                if (sh is not None and k in avg and v.dtype == torch.float32
                        and sh.shape == v.shape):
                    shadows.append(sh); live.append(v.detach())
                else:
                    other[k] = v                          # buffers, non-fp32
            self._plan = (shadows, live, other, avg)
        shadows, live, other, avg = self._plan
        if shadows:
            # sh + (1-d)(v - sh) == sh*d + v*(1-d): the same update, a handful of
            # kernels for all parameters, and exactly v when sh == v (no drift).
            torch._foreach_lerp_(shadows, live, 1.0 - d)
        for k, v in other.items():
            sh = self._shadow.get(k)
            if sh is not None and v.is_floating_point() and k in avg:
                sh.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self._shadow[k] = v.detach().clone().float()
        self._maybe_save()

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


@HOOKS.register_module()
class FiniteGuard(HookBase):
    """Skip a step whose gradients are not finite; stop a run whose weights are not.

    The first 47k-step d768 run went NaN at ~19.4k steps and nothing noticed: the
    job ran to the end, "completed", and its part 2 and a branch cooldown then
    trained from NaN weights. Two defences, both cheap:

    * ``after_backward`` (gradients unscaled, all-reduced, not yet clipped): if
      the total gradient norm is not finite, drop every gradient, so the
      optimizer skips every parameter and its moments are untouched. DDP has
      already averaged the gradients, so every rank takes the same decision.
      ``max_skips`` consecutive skips is a divergence, not a bad batch: raise.
    * ``after_step`` every ``check_every`` steps: if any weight is not finite,
      raise -- the job fails, so an ``afterok`` successor does not start from it.
      Placed before CheckpointSaver, so a non-finite model is not the one saved.
    """

    def __init__(self, check_every=100, max_skips=20):
        self.check_every = int(check_every)
        self.max_skips = int(max_skips)
        self.skipped = 0
        self._run = 0

    def _params(self):
        return [p for p in unwrap_model(self.trainer.model).parameters() if p.requires_grad]

    def _step(self):
        ci = self.trainer.comm_info
        return ci.get("iter", 0) + ci.get("iter_per_epoch", 0) * ci.get("epoch", 0)

    def after_backward(self):
        grads = [p.grad for p in self._params() if p.grad is not None]
        if not grads:
            return
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        if torch.isfinite(norm):
            self._run = 0
            return
        for p in self._params():
            p.grad = None
        self.skipped += 1
        self._run += 1
        it = self._step()
        self.trainer.logger.warning(
            f"FiniteGuard: non-finite gradient norm ({norm.item()}) at iter {it}; step "
            f"skipped ({self.skipped} total, {self._run} in a row)")
        if self._run >= self.max_skips:
            raise RuntimeError(f"FiniteGuard: {self._run} consecutive non-finite steps "
                               f"at iter {it} -- the run has diverged")

    def after_step(self):
        it = self._step() + 1
        if self.check_every <= 0 or it % self.check_every:
            return
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(self._params())))
        if not torch.isfinite(norm):
            raise RuntimeError(f"FiniteGuard: non-finite weights after iter {it} "
                               f"({self.skipped} steps skipped so far) -- stopping before "
                               f"a checkpoint is written from them")
