"""Patches to pimm that this integration needs at import time.

Not "compatibility" in the shim sense — these are live corrections to pimm
behaviour that a helix run depends on, applied when the package is imported.
"""

from __future__ import annotations


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

#: pimm internals this integration reaches for that are NOT public API.
#: Every entry is a name helix actually IMPORTS -- _compat.py:41, trainer.py:15,
#: hooks.py:14 / eval.py:19. Names helix only mentions in prose stay out:
#: asserting one would make `import helix.integrations.pimm` a hard failure over
#: an upstream rename helix does not depend on, which is the opposite of what
#: this gate is for.
_REQUIRED = (
    ("pimm.engines._train_utils", "restore_rng_state"),
    ("pimm.engines.train", "Trainer"),
    ("pimm.engines.hooks.default", "HookBase"),
)


def _assert_pimm_surface():
    """Fail at import, not at step 400,000, if pimm moved under us.

    helix pins pimm by absolute path with no submodule and no dependency
    declaration in pyproject.toml, and monkeypatches one of its privates below.
    A rename there does not raise — `restore_distributed_rng_state` resolves the
    name from module globals at call time, so the patch simply stops applying
    and a preempted run silently reverts to dying on
    `TypeError: RNG state must be a torch.ByteTensor`. provenance.json records
    WHICH pimm produced a run; this is the only thing that checks it still fits.
    """
    import importlib
    missing = []
    for mod, attr in _REQUIRED:
        try:
            if not hasattr(importlib.import_module(mod), attr):
                missing.append(f"{mod}.{attr}")
        except Exception as exc:                     # import error is equally fatal
            missing.append(f"{mod} ({type(exc).__name__}: {exc})")
    if missing:
        raise RuntimeError(
            "helix.integrations.pimm requires pimm internals that are absent: "
            + ", ".join(missing)
            + ". pimm has changed in a way this integration does not survive — "
              "check provenance.json for the commit this run expected.")


_assert_pimm_surface()
