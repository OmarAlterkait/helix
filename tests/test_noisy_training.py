"""Training never reads simulated clean data.

Real detector data has no noise-free counterpart, so a recipe that trains on the
simulated ``coeff_clean`` modality -- as a target, a bin grid, or a validation
reference -- cannot be run on it. Simulation truth is for evaluation only. These
checks pin that: no config loads the clean modality, no training module names
it, the tokenizer's target without clean values is the noisy input (it used to
be zeros, silently), and the production bin grid is derived from noisy values.
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
    from helix.model.tokenize import CoeffTokenize
    assert inspect.signature(CoeffTokenize.__init__).parameters["clean_part"].default is None
    wrapper = ROOT / "helix" / "integrations" / "pimm" / "data.py"
    assert _default_of(wrapper, "CoeffTPCDataset", "modalities") == ("coeff",)


def test_target_without_clean_values_is_the_noisy_input():
    from helix.model.tokenize import PatchConfig, assemble
    from test_tokenize import LENS_T, _rows
    gids = np.array([0, 1, 2, 3, 4, 5])
    band, gid, wire, tau, raw, _, sigma = _rows()
    out = assemble(band, gid, wire, tau, raw, gids=gids, n_wires=np.full(6, 1969, np.int64),
                   band_lengths=LENS_T, norm_sigma=sigma, cfg=PatchConfig(cell_t="grid_center"))
    occ = out["occ"].astype(bool)
    assert occ.any()
    np.testing.assert_array_equal(out["tgt"][occ], out["inp"][occ])


def test_production_bin_grid_is_noisy_and_legacy_tables_rederive_as_clean():
    from helix.data import bins
    assert bins.reference()["params"]["target"] == "noisy"
    assert bins.DEFAULTS["target"] == "noisy"
    legacy = dict(K=128, n_bands=4, events=120, corpus="x")          # a v1-v3 table: no target key
    assert bins.params_of(legacy)["target"] == "clean"
