"""Package boundaries, enforced: what may import pimm-data, what helix and pimm-data duplicate
(pinned so it cannot drift), and that pretraining never names the simulation-only data."""
import os
import re
import subprocess
import sys
import hashlib
import io
import tokenize as pytok
from pathlib import Path

import pytest
import numpy as np

from helix.tpc import readout as hx_geom  # noqa: E402


# ---- the helix / pimm-data boundary ------------------------------------------------------

# The helix/pimm-data boundary, enforced rather than documented.
#
# The invariant: ``helix.core`` and ``helix.tpc`` must NEVER import ``pimm_data``.
# ``helix.data`` and ``helix.integrations`` may, and do.
#
# This is not style. pimm-data requires ``torch>=2.5`` and ``hdf5plugin``
# unconditionally; helix's base install is numpy/h5py/PyWavelets/scipy with torch
# optional. The whole two-container split rests on the asymmetry: the
# corpus-builder image has pywt and torch but no pimm-data, and the training image
# has pimm-data but no pywt. An import added to ``helix.tpc`` would make the DSP
# side undeployable in the image that builds corpora, and nothing else would
# catch it -- the developer's own environment has everything installed.
#
# A source grep cannot do this job: ``readout.py``,
# ``io.py`` and ``noise.py`` all mention pimm_data in comments, and
# ``helix/model/tokenize.py`` documents a registration example. Only an actual
# import in a clean interpreter distinguishes a mention from a dependency.

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Subpackages that must import with pimm_data and torch both absent.
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
         "helix.integrations",
         # reading a run and the window metrics need neither: torch is imported when a
         # model is built, pimm_data when events are read
         "helix.model.checkpoint", "helix.eval", "helix.eval.events"]

#: Modules that are ALLOWED to pull pimm_data in, each with the packages that
#: must be installed for the check to mean anything. Listed so that a module
#: moving across the boundary is a deliberate edit to this file, not a silent
#: change in behaviour.
#:
#: The entries need different environments, which is the split itself: the
#: corpus-builder image can import the dataset module, but only the training
#: image has ``pimm``, so the hooks check runs there and skips here.
#:
#: Each is named by its OWN module path, and the package ``helix.data`` is not
#: among them (see LIGHT_INSIDE_HEAVY).
MAY_IMPORT_PIMM_DATA = [
    ("helix.data.dataset", ["pimm_data"]),
]

#: Modules INSIDE the heavy package that must nonetheless import clean.
#:
#: ``helix.data`` may import pimm_data; these do not need to, and the
#: point of listing them is that Python makes it easy to lose that by accident.
#: A package's ``__init__`` runs before any submodule of it, so an eager
#: re-export in ``helix/data/__init__.py`` puts torch behind ``import
#: helix.data.bins`` without either file changing. It did, and `pimm submit`
#: died on a login node because helix's training config reads the bin table at
#: module scope while pimm's documented launcher environment has "only YAML
#: parsing, Tyro, and Submitit".
#:
#: ``bins`` needs numpy, ``verify`` and ``cells`` h5py -- all base deps, so
#: unlike the list above this one needs no importorskip.
LIGHT_INSIDE_HEAVY = ["helix.data", "helix.data.bins", "helix.data.verify", "helix.data.cells",
                      "helix.data.denoise"]

# helix.integrations.pimm.hooks is deliberately in NEITHER list.
#
# It is ALLOWED to import pimm_data -- it lives in helix.integrations -- but it
# does not do so eagerly: `from helix.data.verify import corpus_identity /
# check_corpus_matches` sit inside the two functions that use them, so a
# module-scope import probe cannot see them and never will. CLEAN is wrong (that
# asserts it must never), and MAY_IMPORT_PIMM_DATA is wrong (that asserts it
# does, eagerly). It was listed in the second, which made the test demand an
# eager import nothing wants.
#
# Found by a clean-room run: this whole module skips wherever pimm is absent,
# which is every environment we normally run the suite in.

_PROBE = (
    "import sys, importlib; importlib.import_module({mod!r}); "
    "print(','.join(sorted({{k.split('.')[0] for k in sys.modules}} "
    "& {{'pimm_data', 'torch'}})))"
)


def _heavy_imports(mod):
    """Import `mod` in a fresh interpreter; return the heavy deps it pulled."""
    r = subprocess.run([sys.executable, "-c", _PROBE.format(mod=mod)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"importing {mod} failed:\n{r.stderr}"
    return set(filter(None, r.stdout.strip().split(",")))


@pytest.mark.parametrize("mod", CLEAN)
def test_stays_clean_of_pimm_data_and_frameworks(mod):
    """The DSP side imports with neither pimm_data nor torch present."""
    pulled = _heavy_imports(mod)
    assert not pulled, (
        f"{mod} pulled in {sorted(pulled)}. helix.core and helix.tpc must import "
        f"with numpy/h5py/PyWavelets/scipy alone -- the corpus-builder container "
        f"has no pimm-data, and torch is an optional extra. If the import is "
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


@pytest.mark.parametrize("mod", LIGHT_INSIDE_HEAVY)
def test_the_light_modules_of_a_heavy_package_stay_light(mod):
    """Corpus metadata is readable wherever the corpus is, launcher included.

    This is the check the DSP-only install cannot make for itself: that install
    has neither torch nor pimm_data, so it cannot tell whether these modules
    merely happen to work or are guaranteed to. Running it in a full
    environment -- where an accidental dependency WOULD resolve -- is what makes
    the guarantee testable.
    """
    pulled = _heavy_imports(mod)
    assert not pulled, (
        f"{mod} pulled in {sorted(pulled)}. It is inside helix.data, which may "
        f"import pimm_data, but it must not: `pimm submit` loads helix's "
        f"training config on the submitting host, and that host is entitled to "
        f"be a launcher-only environment. If the eager import came from "
        f"helix/data/__init__.py, add the name to its _LAZY table instead."
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


#: Importing the PACKAGE helix.tpc only executes its __init__, which pulls only
#: its io module. noise, geometry and the chain are NOT among them,
#: so the CLEAN check above cannot see a violation in any of those files. This
#: sweep imports each submodule on its own.
#:
#: Only pimm_data is forbidden here. torch is NOT: the DSP modules
#: (robust, coherent_gate, noise, ...) import it at
#: module scope by design, and it is a declared optional extra. The invariant this
#: file defends is about pimm-data, not about weight of imports.
@pytest.mark.parametrize("mod", _submodules("helix.core", "helix.tpc"))
def test_no_submodule_of_core_or_tpc_imports_pimm_data(mod):
    r = subprocess.run([sys.executable, "-c", _PROBE.format(mod=mod)],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        missing = re.search(r"No module named '([^']+)'", r.stderr)
        if missing and missing.group(1).split(".")[0] in ("torch",):
            pytest.skip(f"{mod} needs {missing.group(1)}, not installed here")
        pytest.fail(f"importing {mod} failed:\n{r.stderr}")
    pulled = set(filter(None, r.stdout.strip().split(",")))
    assert "pimm_data" not in pulled, (
        f"{mod} imports pimm_data. Only helix.data and helix.integrations may -- "
        f"the corpus-builder container has no pimm-data at all."
    )


# ---- what helix and pimm-data duplicate --------------------------------------------------

# What helix and pimm-data duplicate, pinned so it cannot drift.
#
# Some duplication between the two packages is FORCED by the boundary: helix.core
# and helix.tpc must import with no pimm_data present (docs/ARCHITECTURE.md §2),
# so they cannot import a shared helper, and pimm-data must run with no helix
# present. Where both genuinely need the same thing, both carry it.
#
# The old `tests/test_forward_mirror.py` held those copies together and was deleted
# with the forward-model move -- correctly for the parts that stopped being
# mirrored, WRONGLY for the parts that did not. Geometry never moved: it is still
# duplicated, and from that deletion until this file it was pinned by nothing.
#
# This file can exist because a TEST may import both packages even though
# helix.tpc may not. That is the whole trick, and it is why the pin belongs here
# rather than in either package.
#
# What is deliberately NOT pinned: helix's plane-label parser
# (helix.tpc.readout._label_gid, over helix.tpc.readout.canonical_plane_gid) and
# canonical_plane_id (pimm-data) have ALREADY diverged on purpose -- helix's is
# wire-only and raises on `volume_N_Pixel`, pimm-data's handles it, which is why
# test_canonical_plane_id_stable stayed in pimm-data during the move. A test
# asserting they agree would be wrong.

@pytest.fixture
def pd_geom():
    """pimm-data's geometry module (these tests skip without pimm-data: a TEST may import both
    packages, helix.tpc may not)."""
    pytest.importorskip("torch")
    return pytest.importorskip("pimm_data.geometry")


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _bundled(mod, name="cubic_wireplane_geometry.json"):
    return os.path.join(os.path.dirname(mod.__file__), "data", name)


# ── the detector description ────────────────────────────────────────────────

def test_bundled_geometry_json_is_byte_identical(pd_geom):
    """Both packages ship the same 153 kB JAXTPC export.

    It is an exported artifact, authored by neither, and in both the bundled copy
    is only a fallback (`_resolve` takes any path). But two copies of a detector
    description drift as silently as two copies of code.
    """
    a, b = _bundled(hx_geom), _bundled(pd_geom)
    assert os.path.exists(a) and os.path.exists(b)
    assert _md5(a) == _md5(b), (
        "the bundled plane geometry differs between helix and pimm-data; "
        "densify would size grids from one and the DSP from the other")


def test_both_loaders_agree_on_every_plane(pd_geom):
    """Not just the file -- the parsed registries must match field by field."""
    h = hx_geom.load_plane_registry(_bundled(hx_geom))
    p = pd_geom.load_plane_registry(_bundled(pd_geom))
    assert set(h) == set(p), "different plane ids"
    for gid in sorted(h):
        for key in ("label", "n_wires", "n_ticks", "pedestal"):
            assert h[gid][key] == p[gid][key], f"plane {gid} disagrees on {key}"
        assert np.array_equal(np.asarray(h[gid]["wire_lengths"]),
                              np.asarray(p[gid]["wire_lengths"])), \
            f"plane {gid} disagrees on wire_lengths"


# ── the one duplicated function ─────────────────────────────────────────────


# ---- training is noisy-only --------------------------------------------------------------

# Training never reads simulated clean data or simulation truth.
#
# Real detector data has no noise-free counterpart, so a recipe that trains on the
# simulated ``coeff_clean`` modality -- as a target, a bin grid, or a validation
# reference -- cannot be run on it. The same holds for the ``cells`` modality, the
# simulation truth (helix/data/cells.py). Simulation truth is for downstream tasks and
# evaluation only. These checks pin that: no config loads either modality, no
# training module names them, the tokenizer's target is the noisy input, and the bin
# grid has no clean mode.

ROOT = Path(__file__).resolve().parent.parent


def _code_without_comments(path):
    """Source with comments removed (strings kept: a literal is what loads data)."""
    out = []
    for tok in pytok.generate_tokens(io.StringIO(path.read_text()).readline):
        if tok.type != pytok.COMMENT:
            out.append(tok.string)
    return " ".join(out)


def test_no_config_loads_the_clean_modality():
    hits = [str(p.relative_to(ROOT)) for p in sorted((ROOT / "configs").rglob("*.py"))
            if "coeff_clean" in _code_without_comments(p)]
    assert not hits, f"configs that load simulated clean data: {hits}"


def _string_literals(path):
    import ast
    out = set()
    for tok in pytok.generate_tokens(io.StringIO(path.read_text()).readline):
        if tok.type == pytok.STRING:
            try:
                out.add(ast.literal_eval(tok.string))
            except (ValueError, SyntaxError):          # f-strings
                pass
    return out


#: The downstream configs: supervised on the simulation truth by design (the denoiser).
DOWNSTREAM = ("configs/pimm/denoise.py",)


#: Pretraining code: the configs and the modules that build and train the FM.
def _pretraining_sources():
    return [p for p in sorted((ROOT / "configs").rglob("*.py")) if str(p.relative_to(ROOT)) not in DOWNSTREAM] + \
        sorted((ROOT / "helix" / "model").glob("*.py")) + sorted((ROOT / "helix" / "integrations" / "pimm").glob("*.py"))


def test_no_config_or_training_module_loads_the_truth_modality():
    """``'cells'`` as a string is the modality name ("n_cells" and the like are not). Only the
    downstream configs name it, and they do (so the exemption cannot outlive its reason)."""
    hits = [str(p.relative_to(ROOT)) for p in _pretraining_sources() if "cells" in _string_literals(p)]
    assert not hits, f"pretraining code that names the simulation-truth modality: {hits}"
    assert all("cells" in _string_literals(ROOT / p) for p in DOWNSTREAM)


def test_no_training_module_names_the_clean_modality():
    mods = sorted((ROOT / "helix" / "model").glob("*.py")) + \
        sorted((ROOT / "helix" / "integrations" / "pimm").glob("*.py"))
    hits = [str(p.relative_to(ROOT)) for p in mods if "coeff_clean" in p.read_text()]
    assert not hits, f"training modules that reference coeff_clean: {hits}"


def _default_of(path, cls, arg):
    """A constructor default read from source (parsed, not imported: some of these modules
    import pimm, which the CI image does not carry)."""
    import ast
    tree = ast.parse(path.read_text())
    fn = next(n for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == cls
              for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    names = [a.arg for a in fn.args.args]
    defaults = dict(zip(names[len(names) - len(fn.args.defaults):], fn.args.defaults))
    return ast.literal_eval(defaults[arg])


def test_defaults_load_no_clean_data():
    import inspect
    from helix.model.tokenize import CoeffTokenize, assemble
    assert "clean_part" not in inspect.signature(CoeffTokenize.__init__).parameters
    assert "value_clean" not in inspect.signature(assemble).parameters
    # the pimm wrapper passes its kwargs through, so the default that governs a config is the reader's
    reader = ROOT / "helix" / "data" / "dataset.py"
    assert _default_of(reader, "CoeffTPCDataset", "modalities") == ("coeff",)


def test_target_is_the_noisy_input():
    from helix.model.tokenize import PatchConfig, assemble
    from test_tokenize import LENS_T, _rows
    gids = np.array([0, 1, 2, 3, 4, 5])
    band, gid, wire, tau, raw, sigma = _rows()
    out = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=np.full(6, 1969, np.int64),
                   band_lengths=LENS_T, norm_sigma=sigma, cfg=PatchConfig())
    occ = out["occ"].astype(bool)
    assert occ.any()
    np.testing.assert_array_equal(out["tgt"][occ], out["inp"][occ])


def test_bin_grid_has_no_clean_mode():
    import inspect
    from helix.data import bins
    assert "target" not in inspect.signature(bins.derive).parameters
    assert "target" not in bins.reference()["params"]
