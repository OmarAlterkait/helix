"""Shared test configuration: torch threads, site environment, data-root guards.

``HELIX_REQUIRE_PIMM=1`` makes an unimportable pimm an error and ``HELIX_REQUIRE_DATA=1`` a
missing data root one, instead of silent skips (both once hid dozens of tests behind a green run).
The header naming the site and the data roots reached is printed even under ``-q``: it is what
tells "passed against the real corpus" from "passed against nothing".

Imported as ``_fixtures``, never ``tests._fixtures``: tests/ has no ``__init__.py``, and pimm's
checkout (on PYTHONPATH in the container) ships a regular ``tests`` package that would win.
"""

import importlib.util
import os
import sys

import pytest

# Few intra-op threads: tiny models run faster, and CPU float reductions are deterministic
# (the goldens failed intermittently with the machine's default).
try:
    import torch
    torch.set_num_threads(4)
except ImportError:
    pass


def pytest_configure(config):
    from helix import paths
    paths.apply_site_env()            # HDF5_USE_FILE_LOCKING=FALSE before anything opens a shard
    _require_data()
    if config.get_verbosity() < 0:
        for line in pytest_report_header(config):
            print(line, file=sys.stderr)
    if os.environ.get("HELIX_REQUIRE_PIMM") != "1":
        return
    try:
        if importlib.util.find_spec("pimm") is None:
            raise ImportError("no pimm on sys.path")
        import pimm.datasets.builder  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise pytest.UsageError(f"HELIX_REQUIRE_PIMM=1 but pimm is not importable ({exc}). Put a pimm "
                                f"checkout on PYTHONPATH, or unset HELIX_REQUIRE_PIMM.") from exc


def _roots():
    import _fixtures as tp
    return (("corpus", tp.CORPUS), ("sim", tp.SIM_ROOT), ("pimm", tp.PIMM_ROOT))


def pytest_report_header(config):
    from helix import paths
    roots = _roots()
    present = [n for n, v in roots if v and os.path.isdir(v)]
    absent = [n for n, v in roots if not (v and os.path.isdir(v))]
    return [f"helix site: {paths.site_name() or 'NONE (nothing selected or detected)'}",
            f"helix data: present={','.join(present) or '-'}  absent={','.join(absent) or '-'}"]


def _require_data():
    if os.environ.get("HELIX_REQUIRE_DATA") != "1":
        return
    from helix import paths
    missing = {n: v for n, v in _roots()[:2] if not (v and os.path.isdir(v))}
    if missing:
        detail = "; ".join(f"{k}={v or '(unset)'}" for k, v in missing.items())
        raise pytest.UsageError(f"HELIX_REQUIRE_DATA=1 but these roots do not resolve to a directory: "
                                f"{detail}. site={paths.site_name() or 'none'}. Run `python -m helix.paths`.")
