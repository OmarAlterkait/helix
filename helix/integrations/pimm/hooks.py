"""Run-lifecycle hooks: provenance (+ the corpus-mismatch refusal), the finite guard, the step
watchdog.
"""

from __future__ import annotations

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
      Placed before the checkpoint saver, so a non-finite model is not the one saved.
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


@HOOKS.register_module()
class StepWatchdog(HookBase):
    """Dump every thread's stack and exit if a training step takes longer than ``timeout`` s.

    Three runs hung with no trace: one rank stuck (in a collective, a loader, a save), the
    others waiting in NCCL until its 600 s timeout killed the job with only the collective's
    name. Re-armed after every step -- from the end of step 1, so the first step's torch.compile
    is not timed -- ``faulthandler`` fires on the stuck rank itself, before NCCL's timeout,
    and its stderr shows where it was. No collectives: it cannot hang or desynchronise a run.
    """

    def __init__(self, timeout=300):
        self.timeout = float(timeout)

    def after_step(self):
        import faulthandler
        import sys
        faulthandler.dump_traceback_later(self.timeout, exit=True, file=sys.__stderr__)

    def after_train(self):
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
