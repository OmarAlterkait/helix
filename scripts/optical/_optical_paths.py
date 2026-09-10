"""Where the optical scripts read and write, resolved once.

Every script here carried the same three hardcoded paths, and all three were
wrong for anyone but the author:

* ``sys.path.insert(0, "/sdf/.../helix/scripts/optical")`` — an absolute path
  into a DIFFERENT helix checkout. Running these from this tree silently
  imported that one's modules, the same failure the corpus builder recorded when
  its old ``helix-consolidate`` default meant "32 commits behind, missing the MAD
  median fix". Siblings are importable by being siblings; nothing needs inserting.
* the goop light file, under another user's home directory.
* an output directory inside that other checkout's gitignored ``temp/``.

All three now come from the one table in :mod:`helix.paths`, so a different site
sets two env vars instead of editing ten files.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# helix itself, located from this file — never from a checkout NAME.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helix.paths import root                                   # noqa: E402

#: The goop two-sided PMT light file.
LIGHT = str(root("HELIX_OPTICAL_DATA"))

#: Where figures and JSON land. Under the run-output root, not a sibling
#: checkout's temp/.
OUT = os.environ.get("HELIX_OPTICAL_OUT", str(root("HELIX_EXP") / "optical"))


def ensure_out() -> str:
    """OUT, created. Scripts wrote into a directory that happened to exist."""
    os.makedirs(OUT, exist_ok=True)
    return OUT
