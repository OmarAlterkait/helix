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
