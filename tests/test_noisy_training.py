"""Training never reads simulated clean data or simulation truth.

Real detector data has no noise-free counterpart, so a recipe that trains on the
simulated ``coeff_clean`` modality -- as a target, a bin grid, or a validation
reference -- cannot be run on it. The same holds for the ``cells`` modality, the
simulation truth (helix/data/cells.py). Simulation truth is for downstream tasks and
evaluation only. These checks pin that: no config loads either modality, no
training module names them, the tokenizer's target is the noisy input, and the bin
grid has no clean mode.
"""

import io
import tokenize as pytok
from pathlib import Path

import numpy as np

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


#: Pretraining code: the configs and the modules that build and train the FM.
def _pretraining_sources():
    return sorted((ROOT / "configs").rglob("*.py")) + sorted((ROOT / "helix" / "model").glob("*.py")) + \
        sorted((ROOT / "helix" / "integrations" / "pimm").glob("*.py"))


def test_no_config_or_training_module_loads_the_truth_modality():
    """``'cells'`` as a string is the modality name ("n_cells" and the like are not)."""
    hits = [str(p.relative_to(ROOT)) for p in _pretraining_sources() if "cells" in _string_literals(p)]
    assert not hits, f"pretraining code that names the simulation-truth modality: {hits}"


def test_no_training_module_names_the_clean_modality():
    mods = sorted((ROOT / "helix" / "model").glob("*.py")) + \
        sorted((ROOT / "helix" / "integrations" / "pimm").glob("*.py"))
    hits = [str(p.relative_to(ROOT)) for p in mods if "coeff_clean" in p.read_text()]
    assert not hits, f"training modules that reference coeff_clean: {hits}"


def _default_of(path, cls, arg):
    """A constructor default read from source -- the pimm adapter imports pimm,
    which the CI image does not carry, so it is parsed, not imported."""
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
    wrapper = ROOT / "helix" / "integrations" / "pimm" / "data.py"
    assert _default_of(wrapper, "CoeffTPCDataset", "modalities") == ("coeff",)


def test_target_is_the_noisy_input():
    from helix.model.tokenize import PatchConfig, assemble
    from test_tokenize import LENS_T, _rows
    gids = np.array([0, 1, 2, 3, 4, 5])
    band, gid, wire, tau, raw, sigma = _rows()
    out = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=np.full(6, 1969, np.int64),
                   band_lengths=LENS_T, norm_sigma=sigma, cfg=PatchConfig(cell_t="grid_center"))
    occ = out["occ"].astype(bool)
    assert occ.any()
    np.testing.assert_array_equal(out["tgt"][occ], out["inp"][occ])


def test_bin_grid_has_no_clean_mode():
    import inspect
    from helix.data import bins
    assert "target" not in inspect.signature(bins.derive).parameters
    assert "target" not in bins.reference()["params"]
