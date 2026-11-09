"""What produced a run: the checkouts this process imported, recorded beside its weights.

Pure helix and free of pimm's registries, unlike :mod:`helix.integrations.pimm`, so it is
importable (and testable) wherever helix's suite runs; ``helix.eval`` stamps its results with
the same record.

The module once also generated a ``sys.path`` block that a hook wrote back into pimm's dumped
config, on the belief that pimm's ``scripts/train.sh`` PINNED ``PYTHONPATH`` to its code
snapshot on resume. It prepends the snapshot to the environment's ``PYTHONPATH`` instead, and the
launch recipe's ``setup:`` (``configs/launch/_common.yaml``) exports helix there, so every link
of a chain can import helix without help.
"""

from __future__ import annotations

__all__ = ["provenance", "running_roots"]


def running_roots():
    """``(helix_root, pimm_data_root)`` of the packages in THIS process.

    Taken from the running modules rather than from constants, so the record names
    the code that actually ran -- a link resumed from a different tree included.
    """
    import os

    import helix
    import pimm_data

    def _root(mod):
        return os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))

    return _root(helix), _root(pimm_data)


def provenance():
    """Everything needed to answer "what produced this run directory?".

    Without it the answer is a guess, found by trying configurations until one
    matches. The checkpoint records its own bin tables; this records the code
    around them.
    """
    import os
    import platform
    import sys

    from helix.core.coeff_event import describe_checkout

    helix_root, pimm_data_root = running_roots()
    info = {
        "helix": describe_checkout(helix_root),
        "pimm_data": describe_checkout(pimm_data_root),
        "python": sys.version.split()[0],
        "hostname": platform.node(),
    }
    try:
        import pimm
        info["pimm"] = describe_checkout(
            os.path.dirname(os.path.dirname(os.path.abspath(pimm.__file__))))
    except Exception:
        info["pimm"] = None
    for env in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "APPTAINER_NAME",
                "SINGULARITY_NAME", "SLURM_NTASKS"):
        if os.environ.get(env):
            info.setdefault("env", {})[env] = os.environ[env]
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info
