"""The helix/pimm-data boundary, enforced rather than documented.

The invariant: ``helix.core`` and ``helix.tpc`` must NEVER import ``pimm_data``.
``helix.data`` and ``helix.integrations`` may, and do.

This is not style. pimm-data requires ``torch>=2.5`` and ``hdf5plugin``
unconditionally; helix's base install is numpy/h5py/PyWavelets/scipy with torch
optional. The whole two-container split rests on the asymmetry: the
corpus-builder image has pywt and torch but no pimm-data, and the training image
has pimm-data but no pywt. An import added to ``helix.tpc`` would make the DSP
side undeployable in the image that builds corpora, and nothing else would
catch it -- the developer's own environment has everything installed.

Measured, not assumed: importing ``pimm_data.transform`` (to reach
``@TRANSFORMS.register_module``) pulls 36 pimm_data modules and torch. That
measurement is why the registered transforms live in ``helix/data/transforms.py``
while their kernels stay in ``helix/tpc/``.

A source grep cannot do this job: ``helix/tpc/dense_ops.py``, ``geometry.py``,
``io.py`` and ``noise.py`` all mention pimm_data in comments, and
``helix/model/tokenize.py`` documents a registration example. Only an actual
import in a clean interpreter distinguishes a mention from a dependency.
"""

import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Subpackages that must import with pimm_data, torch and jax all absent.
#:
#: ``helix.integrations`` is in this list on purpose. Its ``__init__`` is a
#: deliberately empty namespace -- "nothing here is imported by helix itself,
#: each module imports a THIRD-PARTY framework at module scope, so importing one
#: is an explicit act by a consumer that already has that framework installed."
#: So the PACKAGE is clean and only its submodules are heavy. That property is
#: load-bearing (it is what keeps ``import helix`` free of torch) and is locked
#: in here; an ``__init__`` that starts re-exporting its submodules would break
#: it silently.
CLEAN = ["helix", "helix.core", "helix.tpc", "helix.model.tokenize",
         "helix.integrations"]

#: Modules that are ALLOWED to pull pimm_data in, each with the packages that
#: must be installed for the check to mean anything. Listed so that a module
#: moving across the boundary is a deliberate edit to this file, not a silent
#: change in behaviour.
#:
#: The two entries need different environments, which is the split itself: the
#: corpus-builder image can import helix.data, but only the training image has
#: ``pimm``, so the hooks check runs there and skips here.
#: helix.data.transforms is listed SEPARATELY from helix.data on purpose.
#: helix/data/__init__.py imports only coeff_reader and coeff_dataset, so
#: importing the package never executes transforms.py -- and transforms.py is
#: reached by STRING through the transform registry, so no other test named it
#: either. It shipped with four broken code paths and both suites stayed green.
#: A module addressed only by string has to be imported somewhere on purpose.
MAY_IMPORT_PIMM_DATA = [
    ("helix.data", ["pimm_data"]),
    ("helix.data.transforms", ["pimm_data", "torch"]),
    ("helix.integrations.pimm.hooks", ["pimm_data", "pimm"]),
]

_PROBE = (
    "import sys, importlib; importlib.import_module({mod!r}); "
    "print(','.join(sorted({{k.split('.')[0] for k in sys.modules}} "
    "& {{'pimm_data', 'torch', 'jax'}})))"
)


def _heavy_imports(mod):
    """Import `mod` in a fresh interpreter; return the heavy deps it pulled."""
    r = subprocess.run([sys.executable, "-c", _PROBE.format(mod=mod)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"importing {mod} failed:\n{r.stderr}"
    return set(filter(None, r.stdout.strip().split(",")))


@pytest.mark.parametrize("mod", CLEAN)
def test_stays_clean_of_pimm_data_and_frameworks(mod):
    """The DSP side imports with none of pimm_data, torch or jax present."""
    pulled = _heavy_imports(mod)
    assert not pulled, (
        f"{mod} pulled in {sorted(pulled)}. helix.core and helix.tpc must import "
        f"with numpy/h5py/PyWavelets/scipy alone -- the corpus-builder container "
        f"has no pimm-data, and torch/jax are optional extras. If the import is "
        f"genuinely needed, the code belongs in helix.data, not here."
    )


def test_source_mentions_are_not_imports():
    """Guard the guard: files that NAME pimm_data must not IMPORT it.

    If this ever passes vacuously -- because the comments were removed -- the
    test above still holds the line. This one exists so the distinction between
    a documented boundary and an enforced one stays visible.
    """
    mentions, imports = [], []
    for sub in ("helix/core", "helix/tpc"):
        d = os.path.join(REPO, sub)
        for root, _, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                with open(path, encoding="utf-8") as fh:
                    src = fh.read()
                if "pimm_data" not in src:
                    continue
                rel = os.path.relpath(path, REPO)
                mentions.append(rel)
                if re.search(r"^\s*(from|import)\s+pimm_data\b", src, re.M):
                    imports.append(rel)
    assert not imports, (
        f"these helix.core/helix.tpc files IMPORT pimm_data: {imports}. "
        f"Only helix.data and helix.integrations may."
    )
    assert mentions, (
        "no helix.core/helix.tpc file mentions pimm_data any more -- if the "
        "boundary comments were deliberately removed, delete this test too"
    )


@pytest.mark.parametrize("mod,needs", MAY_IMPORT_PIMM_DATA,
                         ids=[m for m, _ in MAY_IMPORT_PIMM_DATA])
def test_the_allowed_side_really_does_need_it(mod, needs):
    """The permitted side is permitted because it genuinely depends on it.

    Asserting this keeps MAY_IMPORT_PIMM_DATA honest: a module that stopped
    needing pimm_data should move to CLEAN rather than sit in a list that no
    longer describes it.
    """
    for req in needs:
        pytest.importorskip(req)
    assert "pimm_data" in _heavy_imports(mod), (
        f"{mod} no longer imports pimm_data -- move it to CLEAN"
    )


def _submodules(*subpkgs):
    """Every importable module under the given helix subpackages."""
    out = []
    for sub in subpkgs:
        d = os.path.join(REPO, sub.replace(".", os.sep))
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".py") and not fn.startswith("_"):
                out.append(f"{sub}.{fn[:-3]}")
    return out


#: Importing the PACKAGE helix.tpc only executes its __init__, which pulls 5 of
#: its submodules. noise, dense_ops, geometry and noise_jax are NOT among them,
#: so the CLEAN check above cannot see a violation in any of those files. This
#: sweep imports each submodule on its own.
#:
#: Only pimm_data is forbidden here. torch and jax are NOT: the backend modules
#: (wavelet_ops_torch, coherent_ops_jax, dense_ops, ...) import them at module
#: scope by design, and they are declared optional extras. The invariant this
#: file defends is about pimm-data, not about weight of imports.
@pytest.mark.parametrize("mod", _submodules("helix.core", "helix.tpc"))
def test_no_submodule_of_core_or_tpc_imports_pimm_data(mod):
    r = subprocess.run([sys.executable, "-c", _PROBE.format(mod=mod)],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        missing = re.search(r"No module named '([^']+)'", r.stderr)
        if missing and missing.group(1).split(".")[0] in ("torch", "jax", "jaxlib"):
            pytest.skip(f"{mod} needs {missing.group(1)}, not installed here")
        pytest.fail(f"importing {mod} failed:\n{r.stderr}")
    pulled = set(filter(None, r.stdout.strip().split(",")))
    assert "pimm_data" not in pulled, (
        f"{mod} imports pimm_data. Only helix.data and helix.integrations may -- "
        f"the corpus-builder container has no pimm-data at all."
    )
