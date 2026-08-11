"""The shipped pimm config must define everything pimm's Trainer reads.

A config is only discovered to be incomplete when a launch crashes, usually
after the data has loaded. The first version of `configs/pimm/coeff_fm_encode.py`
defined 8 of the 24 fields the Trainer touches — no `hooks`, no `optimizer`, no
`scheduler`, no `epoch` — and nothing said so, because a config is just a Python
file that executes fine.

This derives the required set from `pimm/engines/train.py` ITSELF (every
`self.cfg.<name>` it mentions) and checks the config against it. Deriving rather
than hardcoding means the test tracks pimm: a field added upstream shows up here
as a failure rather than as a launch-time surprise.

The config is deliberately self-contained (no `_base_`), because `_base_` paths
resolve relative to the config file and inheriting pimm's default_runtime from a
config living in helix would need a brittle cross-repository relative path.
"""

import ast
import os
import pathlib
import re

import pytest

PIMM = "/sdf/group/neutrino/omara/pimm-fm"
TRAIN_PY = os.path.join(PIMM, "pimm/engines/train.py")
DEFAULTS_PY = os.path.join(PIMM, "pimm/engines/defaults.py")
CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "configs", "pimm", "coeff_fm_encode.py")

pimm_src = pytest.mark.skipif(
    not os.path.exists(TRAIN_PY), reason=f"pimm source absent: {TRAIN_PY}")

# Fields the Trainer reads but that are DERIVED by default_config_parser from
# ones a config does set (batch_size / batch_size_val / num_worker divided by
# world size), or that have an explicit getattr default.
DERIVED = {"batch_size_per_gpu", "batch_size_val_per_gpu", "batch_size_test_per_gpu",
           "num_worker_per_gpu", "log_step_offset"}


def _config_names(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    return {t.id for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}


def _trainer_reads(path):
    src = pathlib.Path(path).read_text()
    return set(re.findall(r"self\.cfg\.([a-z_][a-z0-9_]*)", src)) - {"get"}


@pimm_src
def test_config_defines_everything_the_trainer_reads():
    required = _trainer_reads(TRAIN_PY) - DERIVED
    have = _config_names(CONFIG)
    missing = sorted(required - have)
    assert not missing, (
        f"configs/pimm/coeff_fm_encode.py is missing {missing}. pimm's Trainer "
        f"reads these as self.cfg.<name>; a launch would fail on the first one "
        f"it touches. (If a name is newly derived by default_config_parser, add "
        f"it to DERIVED here.)")


@pimm_src
def test_derived_fields_really_are_derived():
    """Guard the escape hatch above: every name in DERIVED must actually be
    assigned by default_config_parser, or we are excusing a real omission."""
    if not os.path.exists(DEFAULTS_PY):
        pytest.skip("pimm/engines/defaults.py absent")
    src = pathlib.Path(DEFAULTS_PY).read_text()
    trainer_src = pathlib.Path(TRAIN_PY).read_text()
    for name in sorted(DERIVED):
        assigned = re.search(rf"cfg\.{name}\s*=", src) or re.search(
            rf'getattr\(\s*self\.cfg,\s*["\']{name}["\']', trainer_src) or re.search(
            rf'self\.cfg\.get\(\s*["\']{name}["\']', trainer_src)
        assert assigned, (
            f"{name!r} is excused as derived, but nothing in default_config_parser "
            f"assigns it and the Trainer has no default for it")


def test_batch_size_is_one():
    """The FM has no event separation, so batch_size > 1 silently trains a model
    whose tokens attend across unrelated events (MULTI_EVENT_BATCHING.md).
    Cheap to assert, and the failure it prevents is invisible."""
    tree = ast.parse(pathlib.Path(CONFIG).read_text())
    vals = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}
    for key in ("batch_size", "batch_size_val", "batch_size_test"):
        assert key in vals, f"config does not set {key}"
        assert isinstance(vals[key], ast.Constant) and vals[key].value == 1, \
            f"{key} must be 1 for the coefficient FM"


def test_config_declares_the_custom_import():
    """The config is the only thing naming the adapter; if this drifts, every
    `type` in it fails to resolve with an unhelpful registry error."""
    src = pathlib.Path(CONFIG).read_text()
    assert "helix.integrations.pimm" in src
    assert "allow_failed_imports=False" in src, (
        "a failed adapter import must be loud — otherwise the first symptom is "
        "'CoeffTokenize is not in the transforms registry'")


@pimm_src
def test_hooks_exclude_the_ones_that_do_not_fit():
    """pimm's default hook list carries SemSegEvaluator, and MAEEvaluator is the
    tempting reuse. Neither fits this model, and both are opt-in — so the config
    must not quietly inherit or add them."""
    src = pathlib.Path(CONFIG).read_text()
    # AST, not substring: the module docstring explains WHY there is no _base_,
    # and a text search happily matches that explanation.
    assert "_base_" not in _config_names(CONFIG), (
        "config gained a _base_; if it inherits pimm's default_runtime it also "
        "inherits SemSegEvaluator in hooks")
    hooks_src = src[src.index("hooks = ["):] if "hooks = [" in src else ""
    for bad in ("SemSegEvaluator", "MAEEvaluator"):
        assert bad not in hooks_src, f"{bad} does not fit the coeff FM"
    assert "CheckpointSaver" in hooks_src, "a run with no checkpointing is a trap"


# ---- the from-scratch categorical path -------------------------------------

def test_categorical_head_from_scratch_needs_explicit_bins(tmp_path):
    """Training from scratch with a categorical head is the ACTUAL plan (m113 is
    out-of-distribution on this corpus), and it needs bin edges that no
    checkpoint supplies.

    The edges are training-set statistics, so the model cannot invent them. This
    pins that the failure is an actionable error at BUILD time rather than an
    assertion on the first forward, and that a bins path satisfies it."""
    import torch
    from helix.model import build_fm

    m = build_fm(dict(n_slot=8, n_band=4, n_plane=6, d=32, blocks=1,
                      dec_blocks=1, heads=4, dec_mode="cross", n_bins=16))
    assert not hasattr(m, "bin_edges"), "a fresh categorical model must have no edges"

    # a bins sidecar in tier1_setup_bins.py's format
    path = tmp_path / "bins.pt"
    torch.save(dict(edges=torch.linspace(-4, 4, 17).repeat(4, 1),
                    cent_asinh=torch.zeros(4, 16), cent_lin=torch.zeros(4, 16),
                    K=16, SIGMA=2.6), path)

    src = pathlib.Path(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "helix", "integrations", "pimm.py")).read_text()
    assert "def _load_bins" in src, "no bins loader in the adapter"
    assert "bins=None" in src, "build_coeff_fm does not accept a bins path"
    assert "TRAINING-SET STATISTICS" in src, (
        "the missing-bins error should explain WHY the model cannot supply them")

    # the loader accepts both accepted shapes
    ns = {}
    exec(compile(src[src.index("def _load_bins"):], "pimm.py", "exec"), {"__name__": "x"}, ns)
    got = ns["_load_bins"](str(path))
    assert "edges" in got and got["edges"].shape == (4, 17)

    conv = tmp_path / "converted.pt"
    torch.save(dict(config={}, state_dict={},
                    bins=dict(edges=torch.linspace(-4, 4, 17).repeat(4, 1))), conv)
    assert "edges" in ns["_load_bins"](str(conv))
