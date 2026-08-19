"""The `sys.path` bootstrap that makes a pimm config self-sufficient.

Kept apart from :mod:`helix.integrations.pimm` on purpose: that module imports
pimm's registries at module scope, and this text generator needs none of them.
Splitting it means the block can be tested wherever helix's suite runs, rather
than only in an image that also has pimm — and the thing under test here is
exactly the piece whose failure mode is invisible until a run tries to resume.

Two packages need bootstrapping, for different reasons:

``helix``      is not installed in the pimm image at all.
``pimm_data``  IS installed, but the image ships 0.3.0, which predates the coeff
               corpus and has no ``CoeffTPCDataset``. So this is not a missing
               import but a STALE one, and the working checkout has to take
               precedence over site-packages.

The block exists because two of pimm's behaviours compose badly:

  * `Config.dump` writes the RESOLVED config dict, so `custom_imports` (a dict)
    survives into `<save_path>/config.py` while a `sys.path` mutation (a
    statement) does not; and
  * `scripts/train.sh` loads that dumped file on resume (line 226) with
    PYTHONPATH pinned to pimm's own code snapshot (line 253), which contains
    neither package.

So the dumped config names modules it cannot import.
:class:`~helix.integrations.pimm.HelixPathBootstrap` writes this block back into
it.
"""

from __future__ import annotations

__all__ = ["BOOTSTRAP_MARK", "bootstrap_block", "has_bootstrap", "running_roots"]

#: Marker identifying a block we wrote, so re-running the hook is a no-op.
BOOTSTRAP_MARK = "# helix: sys.path bootstrap re-added by HelixPathBootstrap"

#: (env var, package) pairs the block resolves, in precedence order.
ROOTS = (("HELIX_ROOT", "helix"), ("PIMM_DATA_SRC", "pimm_data"))


def bootstrap_block(helix_root, pimm_data_root):
    """Source lines putting both checkouts on ``sys.path``, honouring env vars.

    ``insert(1)``, which is neither of the two obvious choices and is the only
    one that works:

    * ``insert(0)`` is defeated by pimm's own loader. ``Config._file2dict`` does
      ``sys.path.insert(0, temp_dir)`` -> ``import_module`` -> ``sys.path.pop(0)``.
      The config executes during that import, so an ``insert(0)`` here lands
      ABOVE the temp dir and the trailing ``pop(0)`` deletes our entry instead of
      pimm's — silently, leaving the temp dir behind.
    * ``append`` survives the pop but loses on precedence: site-packages already
      contains ``pimm_data`` 0.3.0, which would keep winning and take the run
      back to ``No module named 'pimm_data.coeff'``.

    ``insert(1)`` sits just under pimm's temp dir, so the ``pop(0)`` removes the
    temp dir and leaves ours at the front, ahead of site-packages.

    The trailing ``del`` is required, not tidiness. ``Config._file2dict`` keeps
    every module-level name that does not start with ``__``
    (``pimm/utils/config.py:261-262``), so ``_os``/``_sys`` would enter the config
    dict as MODULE OBJECTS; ``Config.dump`` then renders ``_os = <module 'os'
    ...>`` and yapf rejects it with ``YapfError: <unknown>:1:5: invalid syntax``,
    killing the run during setup.
    """
    return (
        f"{BOOTSTRAP_MARK}\n"
        "# `custom_imports` below needs helix importable, and CoeffTPCDataset\n"
        "# needs a pimm_data NEWER than the 0.3.0 in the image. pimm's train.sh\n"
        "# sets PYTHONPATH to its own code snapshot only, so neither is reachable\n"
        "# by environment. insert(1), not insert(0) (pimm's loader pops index 0)\n"
        "# and not append (site-packages' stale pimm_data would still win).\n"
        "import os as _os\n"
        "import sys as _sys\n"
        f"for _v, _p in (('HELIX_ROOT', {str(helix_root)!r}),\n"
        f"               ('PIMM_DATA_SRC', {str(pimm_data_root)!r})):\n"
        "    _p = _os.environ.get(_v) or _p\n"
        "    if _p not in _sys.path:\n"
        "        _sys.path.insert(1, _p)\n"
        "del _os, _sys, _v, _p\n"
    )


def has_bootstrap(src):
    """True if ``src`` already carries a block written by the hook."""
    return BOOTSTRAP_MARK in src


def running_roots():
    """``(helix_root, pimm_data_root)`` of the packages in THIS process.

    Taken from the running modules rather than from constants, so a resumed job
    re-enters the same checkouts job 1 used instead of whatever a default happens
    to point at by then.
    """
    import os

    import helix
    import pimm_data

    def _root(mod):
        return os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))

    return _root(helix), _root(pimm_data)


def describe_checkout(root):
    """``{root, commit, dirty, branch}`` for a git checkout, best-effort.

    Never raises and never blocks: a missing git, a tarball install with no
    ``.git``, or a slow filesystem all degrade to ``commit=None`` rather than
    taking down a training run at step 0.

    ``dirty`` is the field that matters. A commit hash alone says which code was
    COMMITTED, not which code ran; every long run in this project so far was
    launched from a working tree with uncommitted edits, so a hash without a
    dirty flag would have been quietly wrong.
    """
    import os
    import subprocess

    out = {"root": root, "commit": None, "dirty": None, "branch": None}
    if not os.path.isdir(root):
        return out

    def _git(*a):
        try:
            r = subprocess.run(("git",) + a, cwd=root, capture_output=True,
                               text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    # Ask git where the repo is instead of testing for a `.git` DIRECTORY. That
    # test was wrong twice over and produced an all-null record from a real run:
    # helix-extraction's `.git` is a FILE (a gitdir pointer, as any worktree or
    # submodule has), and a src-layout package resolves `root` to `<repo>/src`,
    # which is inside the repo but does not contain `.git` at all. `rev-parse`
    # answers both, and answers "not a repo" by failing.
    top = _git("rev-parse", "--show-toplevel")
    if not top:
        return out
    out["root"] = top
    out["commit"] = _git("rev-parse", "HEAD")
    out["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    st = _git("status", "--porcelain")
    if st is not None:
        out["dirty"] = bool(st)
    return out


def provenance():
    """Everything needed to answer "what produced this run directory?".

    Nothing recorded this before, so for any existing checkpoint the answer is a
    guess — which is how m113 ended up evaluable only by trying configurations
    until one matched. The checkpoint records its own bin tables now; this
    records the code around them.
    """
    import os
    import platform
    import sys

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
