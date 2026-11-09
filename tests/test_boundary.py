"""Package boundaries, enforced: what may import pimm-data and torch, the geometry helix and
pimm-data both ship, and that pretraining never names the simulation truth.

``helix.core`` and ``helix.tpc`` never import ``pimm_data``: pimm-data requires torch>=2.5 and
hdf5plugin, the corpus-builder side does not have it, and a developer's environment, which has
everything, would never notice. Only an import in a clean interpreter tells a dependency from a
mention in a comment, so each check below runs one.
"""
import hashlib
import io
import os
import subprocess
import sys
import tokenize as pytok
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_PROBE = ("import sys, importlib; importlib.import_module({mod!r}); "
          "print(','.join(sorted({{k.split('.')[0] for k in sys.modules}} & {{'pimm_data', 'torch'}})))")


def _heavy_imports(mod):
    """Import ``mod`` in a fresh interpreter: ``(returncode, stderr, {pimm_data, torch} it pulled)``."""
    r = subprocess.run([sys.executable, "-c", _PROBE.format(mod=mod)], capture_output=True, text=True,
                       cwd=ROOT)
    return r.returncode, r.stderr, set(filter(None, r.stdout.strip().split(",")))


def _submodules(*pkgs):
    return [f"{p}.{f.stem}" for p in pkgs for f in sorted((ROOT / p.replace(".", "/")).glob("*.py"))
            if not f.name.startswith("_")]


@pytest.mark.parametrize("mod", _submodules("helix.core", "helix.tpc"))
def test_no_module_of_core_or_tpc_imports_pimm_data(mod):
    """torch is allowed here (the DSP computes with it); pimm_data never is."""
    code, err, pulled = _heavy_imports(mod)
    if code and "No module named 'torch'" in err:
        pytest.skip(f"{mod} needs torch, not installed here")
    assert code == 0, err
    assert "pimm_data" not in pulled, f"{mod} imports pimm_data; only helix.data and helix.integrations may"


#: What must import with neither torch nor pimm_data: the umbrella and the DSP packages, the
#: tokenizer (pimm-data runs it in every DataLoader worker), and everything a training config
#: imports at module scope -- ``pimm submit`` loads the config on a launcher host that has
#: neither (helix.paths, helix.data.verify, helix.data.bins). An eager import in
#: helix/data/__init__.py once put torch behind helix.data.bins and `pimm submit` died on a
#: login node.
LIGHT = ["helix", "helix.core", "helix.tpc", "helix.model.tokenize", "helix.integrations", "helix.paths",
         "helix.data", "helix.data.bins", "helix.data.verify", "helix.data.cells", "helix.data.denoise",
         "helix.model.checkpoint", "helix.eval", "helix.eval.events"]


@pytest.mark.parametrize("mod", LIGHT)
def test_configs_and_the_dsp_import_without_torch_or_pimm_data(mod):
    code, err, pulled = _heavy_imports(mod)
    assert code == 0, err
    assert not pulled, f"{mod} pulled in {sorted(pulled)}"


def test_the_dataset_really_does_need_pimm_data():
    """The permitted side, kept honest: a module that stopped needing it belongs in LIGHT."""
    pytest.importorskip("pimm_data")
    assert "pimm_data" in _heavy_imports("helix.data.dataset")[2]


def test_the_bundled_geometry_is_pimm_datas():
    """Both packages ship the same JAXTPC plane geometry (an exported artifact, authored by
    neither); densify sizes grids from one and the DSP from the other."""
    pytest.importorskip("torch")
    pd_geom = pytest.importorskip("pimm_data.geometry")
    from helix.tpc import readout
    a, b = (os.path.join(os.path.dirname(m.__file__), "data", "cubic_wireplane_geometry.json")
            for m in (readout, pd_geom))
    assert hashlib.md5(open(a, "rb").read()).hexdigest() == hashlib.md5(open(b, "rb").read()).hexdigest()


# ---- pretraining never names the simulation truth ----------------------------------------

#: Supervised on the simulation truth by design.
DOWNSTREAM = ("configs/pimm/denoise.py",)


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


def test_pretraining_reads_no_simulation_truth():
    """Real detector data has neither a noise-free counterpart nor truth, so a recipe that trains
    on either cannot run on it. ``'cells'`` as a string is the truth modality; the modalities the
    dataset knows are the noisy coefficients and that truth (no clean target exists); only the
    downstream configs name it, and they do (so the exemption cannot outlive its reason)."""
    from_src = [p for p in sorted((ROOT / "configs").rglob("*.py")) if str(p.relative_to(ROOT)) not in DOWNSTREAM]
    src = from_src + sorted((ROOT / "helix" / "model").glob("*.py")) + \
        sorted((ROOT / "helix" / "integrations" / "pimm").glob("*.py"))
    hits = [str(p.relative_to(ROOT)) for p in src if "cells" in _string_literals(p)]
    assert not hits, f"pretraining code that names the simulation-truth modality: {hits}"
    assert all("cells" in _string_literals(ROOT / p) for p in DOWNSTREAM)
    assert "VALID_MODALITIES = ('coeff', 'cells')" in (ROOT / "helix" / "data" / "dataset.py").read_text()
