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

from _paths import PIMM_ROOT as PIMM                           # noqa: E402
TRAIN_PY = os.path.join(PIMM, "pimm/engines/train.py")
DEFAULTS_PY = os.path.join(PIMM, "pimm/engines/defaults.py")
CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "configs", "pimm", "coeff_fm_encode.py")

pimm_src = pytest.mark.skipif(
    not os.path.exists(TRAIN_PY), reason=f"pimm source absent: {TRAIN_PY}")

# `pimm_src` only asserts the SOURCE is on disk, which is all the tests that read
# it as text need. Anything that actually imports pimm needs this instead: the
# suite's usual image has the source visible but the package not importable.
import importlib.util as _ilu                                    # noqa: E402
pimm_importable = pytest.mark.skipif(
    _ilu.find_spec("pimm") is None, reason="pimm not importable in this env")

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
    """The FM needs batch_size_per_gpu == 1 — it has no event separation, so a
    rank holding two events would attend across them (MULTI_EVENT_BATCHING.md).

    pimm's `batch_size` is the GLOBAL batch and `default_config_parser` asserts
    `batch_size % world_size == 0`, so the requirement is
    `batch_size = number of GPUs`. The committed config targets a single GPU;
    a multi-GPU launch overrides it (`--options batch_size=<n_gpus>`), which is
    exactly how the research trainer scaled."""
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
    # The buffer EXISTS (persistent, so it rides in the state_dict and a
    # checkpoint no longer needs a sidecar) but is unset — NaN, which cannot be
    # mistaken for data. Absence was the old contract; the sentinel is the new
    # one, and it still makes "built but never given edges" a loud failure
    # rather than a model that silently bucketises everything into one bin.
    import math
    assert hasattr(m, "bin_edges")
    assert not torch.isfinite(m.bin_edges).any(), \
        "a fresh categorical model must have no USABLE edges"

    # a bins sidecar in tier1_setup_bins.py's format
    path = tmp_path / "bins.pt"
    torch.save(dict(edges=torch.linspace(-4, 4, 17).repeat(4, 1),
                    cent_asinh=torch.zeros(4, 16),
                    K=16, SIGMA=2.6), path)

    src = pathlib.Path(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "helix", "integrations", "pimm", "model.py")).read_text()
    assert "def _load_bins" in src, "no bins loader in the adapter"
    assert "bins=None" in src, "build_coeff_fm does not accept a bins path"
    assert "TRAINING-SET STATISTICS" in src, (
        "the missing-bins error should explain WHY the model cannot supply them")

    # Extract the function with ast, not a slice to EOF: the module has other
    # classes whose decorators reference pimm names, and a slice would drag them
    # in. (The same fragility that once truncated the model extraction.) Less
    # acute since the split put `_load_bins` in a 90-line file, but the reason
    # the slice was wrong has not changed.
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == "_load_bins")
    ns = {"torch": torch}
    exec(compile(ast.get_source_segment(src, node), "model.py", "exec"), ns)
    got = ns["_load_bins"](str(path))
    assert "edges" in got and got["edges"].shape == (4, 17)

    conv = tmp_path / "converted.pt"
    torch.save(dict(config={}, state_dict={},
                    bins=dict(edges=torch.linspace(-4, 4, 17).repeat(4, 1))), conv)
    assert "edges" in ns["_load_bins"](str(conv))


def test_configs_bootstrap_helix_onto_sys_path():
    """Every pimm config must put helix on sys.path ITSELF, by APPEND.

    pimm's scripts/train.sh hard-sets PYTHONPATH to its own code directory in
    every branch, so a launch through `pimm submit` cannot see helix by
    environment alone. The config is a Python file executed before
    `custom_imports` is processed (Config.fromfile), so it can bootstrap itself
    — which avoids installing helix into a shared image.

    It must use insert(1) — the only position that works, as both obvious
    choices fail in opposite directions:

      insert(0)  pimm's loader does insert(0, temp_dir) -> import -> pop(0), and
                 the config executes during that import, so an insert(0) here is
                 what the pop deletes. Fails silently: helix stays unimportable
                 and custom_imports raises a bare ImportError with the real cause
                 swallowed by import_modules_from_strings.
      append     survives the pop, but loses to site-packages — the pimm image
                 ships pimm_data 0.3.0, which has no coeff module, so the run
                 dies on "No module named 'pimm_data.coeff'".
    """
    import re
    from pathlib import Path

    cfg_dir = Path(__file__).resolve().parent.parent / "configs" / "pimm"
    found = sorted(cfg_dir.glob("coeff_fm_*.py"))
    assert found, "no pimm configs found"
    for path in found:
        src = path.read_text()
        # The ASSIGNMENT, not the word — the bootstrap comment mentions
        # custom_imports and would otherwise match first, making the ordering
        # check below compare against the wrong position.
        assign = src.index("custom_imports = ")
        # Ignore comment lines: they discuss insert(0) and append by name.
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.lstrip().startswith("#"))
        boot = re.search(r"_sys\.path\.(insert|append)\(\s*(\d+)?", code)
        assert boot, f"{path.name} does not bootstrap onto sys.path"
        assert (boot.group(1), boot.group(2)) == ("insert", "1"), (
            f"{path.name} uses sys.path.{boot.group(1)}({boot.group(2) or ''}) — "
            f"must be insert(1): insert(0) is removed by pimm's own pop(0), and "
            f"append loses to the stale pimm_data in site-packages")
        assert src.index("_sys.path.") < assign, (
            f"{path.name} bootstraps AFTER custom_imports, which is too late")
        assert "PIMM_DATA_SRC" in src, (
            f"{path.name} bootstraps helix but not pimm_data; the image's 0.3.0 "
            f"has no CoeffTPCDataset")


def test_bootstrap_configs_also_register_the_rewrite_hook():
    """A config that bootstraps sys.path MUST also list HelixPathBootstrap.

    The two are one mechanism split across a process boundary: the source-level
    bootstrap gets job 1 running, and the hook puts that bootstrap back into the
    config pimm DUMPS so job 2 (which loads the dump, not this file) can import
    helix at all. Having only the first is the dangerous state — it works
    perfectly until the first requeue, hours in.
    """
    from pathlib import Path

    cfg_dir = Path(__file__).resolve().parent.parent / "configs" / "pimm"
    for path in sorted(cfg_dir.glob("coeff_fm_*.py")):
        src = path.read_text()
        if "_sys.path." not in src:
            continue
        assert "HelixPathBootstrap" in src, (
            f"{path.name} bootstraps sys.path but never registers "
            f"HelixPathBootstrap, so a resumed job cannot import helix")


def test_bootstrap_block_is_valid_python_that_appends():
    """The block the hook writes must parse, and must append rather than insert."""
    import ast

    from helix.integrations._bootstrap import bootstrap_block

    block = bootstrap_block("/some/checkout", "/some/pimm-data/src")
    ast.parse(block)                      # a syntax error here is unresumable
    # The block's own comments explain the insert(0) trap, so check the CODE.
    code = "\n".join(ln for ln in block.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "_sys.path.insert(1, _p)" in code, (
        "insert(1) is the only workable position: insert(0) is deleted by pimm's "
        "own sys.path.pop(0), and append loses to site-packages' stale pimm_data")
    assert "insert(0" not in code
    assert ".append(" not in code
    assert "'/some/checkout'" in block
    assert "'/some/pimm-data/src'" in block, "pimm_data must be bootstrapped too"
    assert "HELIX_ROOT" in block and "PIMM_DATA_SRC" in block


def test_rewritten_config_is_importable_without_helix_on_the_path(tmp_path):
    """End-to-end: dump-shaped config + the hook's block -> helix imports.

    Simulates what train.sh's resume branch does — load <save_path>/config.py in
    a process whose sys.path does NOT contain helix — and asserts the rewritten
    file repairs it.
    """
    import subprocess
    import sys
    from pathlib import Path

    from helix.integrations._bootstrap import bootstrap_block

    root = str(Path(__file__).resolve().parent.parent)

    # A "fresh" pimm_data the block points at, and a "stale" one standing in for
    # the 0.3.0 in site-packages. Only the fresh one has CoeffTPCDataset, which is
    # exactly the difference that broke the first real launch.
    fresh = tmp_path / "fresh"
    (fresh / "pimm_data").mkdir(parents=True)
    (fresh / "pimm_data" / "__init__.py").write_text(
        "CoeffTPCDataset = object\nWHICH = 'fresh'\n")
    stale = tmp_path / "stale"
    (stale / "pimm_data").mkdir(parents=True)
    (stale / "pimm_data" / "__init__.py").write_text("WHICH = 'stale'\n")

    dumped = "custom_imports = dict(imports=['helix.integrations.pimm'])\n"
    cfg = tmp_path / "config.py"
    cfg.write_text(bootstrap_block(root, str(fresh)) + "\n" + dumped)

    # Execute it the way the loader does: helix absent, and the stale pimm_data
    # APPENDED, which is where site-packages sits relative to a fresh insert.
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import runpy, sys\n"
        f"sys.path[:] = [p for p in sys.path if {root!r} not in p]\n"
        f"sys.path.append({str(stale)!r})\n"
        f"runpy.run_path({str(cfg)!r})\n"
        "import importlib\n"
        "importlib.import_module('helix')\n"
        "pd = importlib.import_module('pimm_data')\n"
        "print('HELIX-IMPORTABLE', 'pimm_data=' + pd.WHICH)\n")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, str(probe)], capture_output=True,
                         text=True, env=env)
    assert "HELIX-IMPORTABLE" in out.stdout, out.stderr[-2000:]
    assert "pimm_data=fresh" in out.stdout, (
        "the bootstrapped pimm_data lost to the one later on sys.path — this is "
        f"the 'No module named pimm_data.coeff' failure. stdout={out.stdout!r}")


def test_configs_leave_no_module_objects_in_the_namespace():
    """The sys.path bootstrap must not leak `_os`/`_sys` into the config dict.

    `Config._file2dict` keeps every module-level name not starting with `__`
    (pimm/utils/config.py:261-262). A leaked module object reaches `Config.dump`,
    which renders `_os = <module 'os' ...>` and dies in yapf with
    `YapfError: <unknown>:1:5: invalid syntax` — during setup, so the run never
    starts. Encodes pimm's filter exactly rather than trusting the `del`.
    """
    import types
    from pathlib import Path

    cfg_dir = Path(__file__).resolve().parent.parent / "configs" / "pimm"
    for path in sorted(cfg_dir.glob("coeff_fm_*.py")):
        ns = {}
        exec(compile(path.read_text(), str(path), "exec"), ns)
        leaked = sorted(k for k, v in ns.items()
                        if not k.startswith("__") and isinstance(v, types.ModuleType))
        assert not leaked, (
            f"{path.name} leaks module objects {leaked} into the config dict; "
            f"Config.dump cannot serialise them and the run dies at setup")


def test_bootstrap_block_deletes_its_temporaries():
    """Same guarantee for the block the hook writes into the dumped config."""
    import types

    from helix.integrations._bootstrap import bootstrap_block

    ns = {}
    exec(compile(bootstrap_block("/a", "/b"), "<block>", "exec"), ns)
    leaked = sorted(k for k, v in ns.items()
                    if not k.startswith("__") and isinstance(v, types.ModuleType))
    assert not leaked, f"bootstrap_block leaks {leaked}"


@pimm_importable
def test_rng_restore_shim_moves_state_back_to_cpu():
    """pimm's resume must survive pimm's own checkpoint loader.

    `checkpoints.py:940` loads with `map_location=lambda storage, loc:
    storage.cuda()`, which moves EVERY tensor in the payload to the GPU --
    including the saved RNG state. `torch.set_rng_state` and
    `torch.cuda.set_rng_state_all` both require CPU ByteTensors, so `resume=True`
    raised "TypeError: RNG state must be a torch.ByteTensor" before the first
    step. A preempted run could then only warm-start from the weights, resetting
    the optimizer and global_step -- which restarts the LR schedule on every
    eviction and makes a preemptable queue unusable for a long run.

    Importing the adapter installs the fix; this pins that it survives a CUDA-ish
    payload and stays idempotent.
    """
    import torch

    import helix.integrations.pimm  # noqa: F401  (installs the shim on import)
    from pimm.engines import _train_utils as tu

    assert getattr(tu.restore_rng_state, "_helix_cpu_shim", False), \
        "importing helix.integrations.pimm did not install the RNG shim"

    # a payload shaped like pimm's, with the RNG state on a non-CPU-like device
    class _FakeDev:
        type = "cuda"

    class _FakeTensor:
        def __init__(self, real):
            self._real = real
            self.device = _FakeDev()

        def cpu(self):
            return self._real

    real = torch.get_rng_state()
    state = {"python": __import__("random").getstate(),
             "numpy": __import__("numpy").random.get_state(),
             "torch": _FakeTensor(real)}
    tu.restore_rng_state(state)          # must not raise
    assert torch.equal(torch.get_rng_state(), real)

    # idempotent: importing again must not double-wrap
    first = tu.restore_rng_state
    from helix.integrations.pimm import _patch_rng_restore_to_cpu
    _patch_rng_restore_to_cpu()
    assert tu.restore_rng_state is first
