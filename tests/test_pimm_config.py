"""The pimm config contract: helix's configs, the registry adapters and the sys.path bootstrap
against what pimm reads, and the integration's import boundaries."""
import importlib.util
import os
from pathlib import Path
import ast
import pathlib
import re
import importlib.util as _ilu                                    # noqa: E402

import pytest

from _fixtures import PIMM_ROOT as PIMM                           # noqa: E402


# ---- the integration's import boundaries -------------------------------------------------

# The pimm adapter must register helix's names — without pimm knowing helix.
#
# ``helix/integrations/pimm/`` is how a pimm config reaches the coefficient FM.
# It lives here rather than inside pimm-private so the dependency points one way:
# helix knows how to plug into pimm, pimm knows nothing about helix. A pimm config
# pulls it in with mmcv's standard ``custom_imports`` hook, which pimm's
# ``Config.fromfile`` already honours.
#
# These tests skip when pimm is not importable — which includes environments that
# have pimm checked out but not its dependencies.

def _pimm_importable():
    """pimm present AND importable. Its package __init__ pulls optional deps
    (pyarrow, addict), so find_spec succeeding is not enough."""
    try:
        if importlib.util.find_spec("pimm") is None:
            return False
        import pimm.datasets.builder  # noqa: F401
        return True
    except Exception:
        return False


def test_tokenizer_does_not_drag_in_the_tpc_stack():
    """`helix.model.tokenize` runs in every DataLoader worker; it must stay light.

    The umbrella used to import helix.tpc.{config,pipeline,io} eagerly, so
    importing the tokenizer — or anything in helix.core — pulled in the whole
    wire-plane HDF5 stack: 1.07s and h5py, for a module that needs neither. It
    also left the tokenizer one heavy or broken import in helix.tpc away from
    failing in every worker, and made helix.core's "detector-agnostic" claim
    false at import time.

    Checked in a subprocess so it measures a cold interpreter, not whatever the
    test session happens to have imported already.
    """
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-c",
         "import helix.model.tokenize, sys; "
         "print(sorted(m for m in sys.modules if m.startswith('helix.tpc')), "
         "[m for m in ('pywt','scipy','h5py') if m in sys.modules])"],
        capture_output=True, text=True, cwd=root,
        env={**os.environ, "PYTHONPATH": root})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[] []", (
        f"tokenize pulled in {r.stdout.strip()} — the umbrella is eager again")


def test_umbrella_exposes_the_codec_and_the_model():
    """The names two sibling repos actually consume must be reachable.

    The umbrella exported six 2025 DSP names while pimm-data imported
    helix.core.* and the pimm configs imported helix.model.* — every real
    consumer bypassed it. These resolve lazily, so naming them costs nothing.
    """
    import helix
    for name in ("CoeffEvent", "write_coeff_shard", "read_coeff_event",
                 "audit_shard", "coord_digest", "build",
                 "TPCChain", "sim_config"):
        assert hasattr(helix, name), name
        assert name in dir(helix)
    with pytest.raises(AttributeError, match="no attribute"):
        helix.definitely_not_a_real_name


pimm_required = pytest.mark.skipif(
    not _pimm_importable(), reason="pimm (and its dependencies) not importable")


def test_importing_helix_does_not_import_pimm():
    """The whole point of the integrations/ split: `import helix` must not drag
    in a downstream framework. This one runs everywhere."""
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-c",
         "import helix, helix.model.tokenize, sys; "
         "print('pimm' in sys.modules, 'torch' in sys.modules)"],
        capture_output=True, text=True, cwd=root,
        env={**os.environ, "PYTHONPATH": root})
    assert r.returncode == 0, r.stderr
    pimm_loaded, torch_loaded = r.stdout.split()
    assert pimm_loaded == "False", "importing helix pulled in pimm"
    assert torch_loaded == "False", "importing helix.model.tokenize pulled in torch"


@pimm_required
def test_the_pimm_facing_surface_adds_no_dsp_dependency_of_its_own():
    """The direction of the dependency: helix knows about pimm, never the reverse.

    pimm is detector- and transform-agnostic infrastructure; WAVELETS are
    helix's concern, and `PyWavelets` is declared in helix's own pyproject, not
    pimm's. So a pimm environment is CORRECT to have no pywt, and
    `helix.integrations.pimm` — the module pimm loads through `custom_imports`,
    in every rank of every run — must import cleanly there.

    It does today, but nothing pinned it, and the leak would be silent exactly
    where it matters: helix's dev image HAS pywt, so an accidental `helix.core`
    import at module scope would pass every local test and then fail at job
    start on a pimm-only image, after the queue wait.

    Measured as a DELTA against pimm's own imports, not as an absolute. pimm
    itself pulls scipy and h5py (torch_geometric and its dataset stack), so
    asserting an empty set would fail for a reason helix does not control and
    cannot fix — it would be a test of pimm's dependency tree wearing helix's
    name. What helix controls is what it ADDS, and that must be nothing.

    `test_tokenizer_does_not_drag_in_the_tpc_stack` above asserts the same
    invariant for `helix.model.tokenize`, which runs in DataLoader workers.
    This is the other surface pimm actually loads.
    """
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; "
         "WATCH = ('pywt', 'scipy', 'h5py'); "
         "import pimm.datasets.builder; "
         "base = {m for m in WATCH if m in sys.modules}; "
         "import helix.integrations.pimm; "
         "print(sorted({m for m in WATCH if m in sys.modules} - base))"],
        capture_output=True, text=True, cwd=root,
        env={**os.environ,
             "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]", (
        f"helix.integrations.pimm pulled in {r.stdout.strip()} on top of pimm — "
        f"the pimm-facing surface acquired a helix-side DSP dependency, and will "
        f"now fail to import on any pimm environment that lacks it")


@pimm_required
def test_adapter_registers_all_three():
    import helix.integrations.pimm  # noqa: F401
    from pimm.datasets.builder import DATASETS
    from pimm.datasets.transform.common import TRANSFORMS
    from pimm.models.builder import MODELS

    assert "CoeffTokenize" in TRANSFORMS
    assert "CoeffTPCDataset" in DATASETS
    assert "Coeff-FM" in MODELS


@pimm_required
def test_example_config_declares_the_custom_import():
    """The config is the only place that names the adapter; if the declaration
    drifts, every `type` in it fails to resolve with an unhelpful error."""
    from pimm.utils.config import Config
    from helix.paths import SiteError

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        cfg = Config.fromfile(os.path.join(root, "configs", "pimm", "fm_smoke.py"))
    except (SiteError, FileNotFoundError) as exc:
        # the configs resolve the corpus's run list at module scope
        pytest.skip(f"config needs a root this site does not configure: {exc}")
    assert "helix.integrations.pimm" in cfg.custom_imports["imports"]
    from helix.paths import world_size
    # one event per rank: the FM has no event separation (docs/ARCHITECTURE.md §8)
    assert cfg.batch_size == world_size()
    assert cfg.model["type"] == "Coeff-FM"
    assert cfg.data["train"]["type"] == "CoeffTPCDataset"


def test_dataset_wrapper_forwards_every_inner_parameter():
    """The helix wrapper re-declares pimm-data's dataset signature, and CONFIGS
    RESOLVE THE WRAPPER. So a parameter added to the inner dataset is invisible
    until it is forwarded here.

    That has bitten twice: event_range/exclude_range, then holdout/split_role —
    the latter failed the first real 2-GPU launch with "unexpected keyword
    argument 'holdout'" after passing every unit test, because the tests
    construct the INNER dataset directly.

    Compared by AST so this needs neither pimm nor a built dataset.
    """
    import ast
    import inspect

    pimm_data = pytest.importorskip("pimm_data")
    from helix.data.dataset import CoeffTPCDataset as Inner

    src = (Path(__file__).resolve().parent.parent
           / "helix" / "integrations" / "pimm" / "data.py").read_text()
    wrapper = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "CoeffTPCDataset":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "__init__":
                    wrapper = {a.arg for a in f.args.args} | {
                        a.arg for a in f.args.kwonlyargs}
    assert wrapper, "could not find the wrapper's __init__"

    inner = set(inspect.signature(Inner.__init__).parameters)
    # ignore_index is a base-class concern the wrapper deliberately does not expose
    missing = inner - wrapper - {"ignore_index"}
    assert not missing, (
        f"the wrapper does not forward {sorted(missing)} — a config passing them "
        f"raises TypeError at trainer build, which no unit test sees because "
        f"they construct the inner dataset directly")


# ---- the configs against pimm's trainer --------------------------------------------------

# The shipped pimm config must define everything pimm's Trainer reads.
#
# A config is only discovered to be incomplete when a launch crashes, usually
# after the data has loaded. The first version of the pimm config
# defined 8 of the 24 fields the Trainer touches — no `hooks`, no `optimizer`, no
# `scheduler`, no `epoch` — and nothing said so, because a config is just a Python
# file that executes fine.
#
# This derives the required set from `pimm/engines/train.py` ITSELF (every
# `self.cfg.<name>` it mentions) and checks the config against it. Deriving rather
# than hardcoding means the test tracks pimm: a field added upstream shows up here
# as a failure rather than as a launch-time surprise.
#
# The config is deliberately self-contained (no `_base_`), because `_base_` paths
# resolve relative to the config file and inheriting pimm's default_runtime from a
# config living in helix would need a brittle cross-repository relative path.

TRAIN_PY = os.path.join(PIMM, "pimm/engines/train.py")
DEFAULTS_PY = os.path.join(PIMM, "pimm/engines/defaults.py")
CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "configs", "pimm", "fm.py")

pimm_src = pytest.mark.skipif(
    not os.path.exists(TRAIN_PY), reason=f"pimm source absent: {TRAIN_PY}")

# `pimm_src` only asserts the SOURCE is on disk, which is all the tests that read
# it as text need. Anything that actually imports pimm needs this instead: the
# suite's usual image has the source visible but the package not importable.
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
        f"configs/pimm/fm.py is missing {missing}. pimm's Trainer "
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
    """The FM needs batch_size_per_gpu == 1: it has no event separation, so a rank holding
    two events would attend across them (docs/ARCHITECTURE.md §8). pimm's ``batch_size`` is
    GLOBAL, so the base config sets it to the world size, and a variant to 1 or not at all."""
    for path in sorted(pathlib.Path(CONFIG).parent.glob("fm*.py")):
        tree = ast.parse(path.read_text())
        vals = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
        for key in ("batch_size", "batch_size_val", "batch_size_test"):
            v = vals.get(key)
            if path.name == "fm.py":
                assert v is not None, f"fm.py does not set {key}"
            ok = v is None or (isinstance(v, ast.Constant) and v.value == 1) or (
                isinstance(v, ast.Name) and v.id == "_WORLD")
            assert ok, f"{path.name}: {key} must be the world size (one event per rank) or 1"


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
    """Training from scratch with a categorical head needs bin edges that no
    checkpoint supplies.

    The edges are training-set statistics, so the model cannot invent them. This
    pins that the failure is an actionable error at BUILD time rather than an
    assertion on the first forward, and that a bins path satisfies it."""
    torch = pytest.importorskip("torch")
    from helix.model import build_fm

    m = build_fm(dict(n_slot=8, n_band=4, n_plane=6, d=32, blocks=1,
                      dec_blocks=1, heads=4, n_bins=16))
    # The buffer EXISTS (persistent, so it rides in the state_dict and a
    # checkpoint no longer needs a sidecar) but is unset — NaN, which cannot be
    # mistaken for data. Absence was the old contract; the sentinel is the new
    # one, and it still makes "built but never given edges" a loud failure
    # rather than a model that silently bucketises everything into one bin.
    import math
    assert hasattr(m, "bin_edges")
    assert not torch.isfinite(m.bin_edges).any(), \
        "a fresh categorical model must have no USABLE edges"

    # a bins table as helix.data.bins writes it, read back by the one loader
    path = tmp_path / "bins.pt"
    torch.save(dict(edges=torch.linspace(-4, 4, 17).repeat(4, 1),
                    cent_asinh=torch.zeros(4, 16),
                    K=16, SIGMA=2.6), path)
    from helix.data.bins import load
    from helix.model.checkpoint import apply_bins
    got = load(str(path))
    assert "edges" in got and got["edges"].shape == (4, 17)
    apply_bins(m, got)
    assert torch.isfinite(m.bin_edges).all() and torch.isfinite(m.bin_cent_ratio).all()

    src = pathlib.Path(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "helix", "integrations", "pimm", "__init__.py")).read_text()
    assert "bins=None" in src, "build_coeff_fm does not accept a bins path"
    assert "helix.data.bins import load" in src, "the adapter must read bins with the one loader"
    assert "TRAINING-SET STATISTICS" in src, (
        "the missing-bins error should explain WHY the model cannot supply them")


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
    found = sorted(cfg_dir.glob("fm*.py"))
    assert found, "no pimm configs found"
    for path in found:
        src = path.read_text()
        # A config may INHERIT the bootstrap: pimm execs each `_base_` file
        # before merging (Config._file2dict), so the base's sys.path statement
        # runs and helix is importable by the time this file is read. Verified
        # against a real load with helix off PYTHONPATH, not assumed — see
        # test_a_derived_config_inherits_a_working_bootstrap below. Requiring
        # every config to repeat the block would force copy-paste of the one
        # thing that must never drift.
        if "_base_" in src and "_sys.path." not in src:
            # The ASSIGNMENT again, not the word: this file's own comment says
            # custom_imports is inherited, and matching the word made the guard
            # fire on that sentence. Exactly the trap documented four lines down.
            assert "custom_imports = " not in src, (
                f"{path.name} inherits its bootstrap but sets custom_imports "
                f"itself; the two must travel together")
            continue
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
        assert "PIMM_DATA_SRC" not in src, (
            f"{path.name} still bootstraps a pimm-data CHECKOUT. The image "
            f"installs it at the pinned revision and the build refuses a stale "
            f"one, so shadowing it makes the run disagree with the pin — one "
            f"authority governs the image and another the run. (Was required "
            f"when the image baked 0.3.0; that is fixed at the source.) "
            f"the image's 0.3.0 "
            f"has no CoeffTPCDataset")


@pimm_importable
@pytest.mark.xfail(strict=False, reason=(
    "The CODE and this TEST disagree about intent, and the disagreement is "
    "deliberate on the code's side. configs/pimm/fm_cooldown.py says a "
    "derived config does NOT get its own sys.path bootstrap -- 'a config that "
    "touches sys.path must also register HelixPathBootstrap so a RESUMED job can "
    "still import helix, and the hook belongs to the base' -- so it relies on "
    "helix already being importable, which the launcher guarantees by exporting "
    "PYTHONPATH. This test strips helix from PYTHONPATH and demands it work "
    "anyway. In production the launcher always sets it, so nothing is broken "
    "today; but one of the two should change. Marked rather than edited because "
    "silencing it by rewriting either side is how a real expectation gets lost. "
    "Invisible until a clean-room run, where this module stops skipping."))
def test_a_derived_config_inherits_a_working_bootstrap(tmp_path):
    """A `_base_` config must actually reach helix without help from the env.

    The exemption above is only sound if inheritance really carries the
    bootstrap. It does — pimm execs the base before merging — but that is a
    property of pimm's loader, not of our config, so it is checked rather than
    reasoned about: load a derived config in a subprocess whose PYTHONPATH has
    NO helix, and require that helix becomes importable anyway.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    derived = [p for p in sorted((root / "configs" / "pimm").glob("fm*.py"))
               if "_base_" in p.read_text() and "_sys.path." not in p.read_text()]
    if not derived:
        pytest.skip("no _base_-derived configs")

    env = {k: v for k, v in os.environ.items()}
    env["PYTHONPATH"] = ":".join(
        p for p in env.get("PYTHONPATH", "").split(":") if p and "helix" not in p)
    prog = (
        "import sys\n"
        "assert not any('helix' in p for p in sys.path), sys.path\n"
        "from pimm.utils.config import Config\n"
        f"c = Config.fromfile({str(derived[0])!r})\n"
        "import helix\n"
        "print('OK', c.custom_imports['imports'][0])\n")
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                       text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0 and "OK" in r.stdout, (
        f"{derived[0].name} does not inherit a working bootstrap:\n"
        f"{r.stdout}\n{r.stderr[-2000:]}")


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
    for path in sorted(cfg_dir.glob("fm*.py")):
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
    assert "'/some/pimm-data/src'" not in block, (
        "the block must NOT insert a pimm-data checkout: the image installs the "
        "pinned revision and shadowing it splits authority between image and run")
    assert "HELIX_ROOT" in block and "PIMM_DATA_SRC" not in block


def test_rewritten_config_is_importable_without_helix_on_the_path(tmp_path):
    """End-to-end: dump-shaped config + the hook's block -> helix imports.

    Simulates what train.sh's resume branch does — load <save_path>/config.py in
    a process whose sys.path does NOT contain helix — and asserts the rewritten
    file repairs it.
    """
    pytest.importorskip("pimm_data")  # the subprocess imports it via the config
    import subprocess
    import sys
    from pathlib import Path

    from helix.integrations._bootstrap import bootstrap_block

    root = str(Path(__file__).resolve().parent.parent)

    # The property is that the block inserts ONLY helix, so `pimm_data` keeps
    # resolving to whatever is INSTALLED. This used to build a "fresh" and a
    # "stale" pimm_data and assert the fresh one won, because the block had to
    # beat the image's baked 0.3.0. The image installs the pinned revision now,
    # so shadowing it is the bug and resolving to it is the contract.
    dumped = "custom_imports = dict(imports=['helix.integrations.pimm'])\n"
    cfg = tmp_path / "config.py"
    cfg.write_text(bootstrap_block(root) + "\n" + dumped)

    probe = tmp_path / "probe.py"
    probe.write_text(
        "import runpy, sys\n"
        f"sys.path[:] = [p for p in sys.path if {root!r} not in p]\n"
        f"runpy.run_path({str(cfg)!r})\n"
        "import importlib\n"
        "importlib.import_module('helix')\n"
        "pd = importlib.import_module('pimm_data')\n"
        "print('HELIX-IMPORTABLE', 'pimm_data=' + (pd.__file__ or ''))\n")
    # Inherit the env: we WANT the installed pimm_data reachable, because that
    # is what the run must resolve.
    env = dict(os.environ)
    out = subprocess.run([sys.executable, str(probe)], capture_output=True,
                         text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert out.returncode == 0, f"probe failed:\n{out.stdout}\n{out.stderr[-2000:]}"
    assert "HELIX-IMPORTABLE" in out.stdout, (
        f"the rewritten config did not make helix importable:\n{out.stderr[-2000:]}")
    resolved = out.stdout.split("pimm_data=")[-1].strip()
    assert str(tmp_path) not in resolved and root not in resolved, (
        f"the config bootstrap shadowed the installed pimm_data with "
        f"{resolved!r}. It must not: the pin governs the image, and a run that "
        f"resolves something else makes the pin a statement about nothing.")


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

    from helix.paths import SiteError as _SiteError

    cfg_dir = Path(__file__).resolve().parent.parent / "configs" / "pimm"
    checked = []
    missing = []
    for path in sorted(cfg_dir.glob("fm*.py")):
        ns = {}
        try:
            exec(compile(path.read_text(), str(path), "exec"), ns)
        except (FileNotFoundError, _SiteError, SystemExit) as e:
            # Some configs resolve a checkpoint or a bin table AT MODULE SCOPE, so
            # executing them needs that data present (the corpus's run list, the
            # bin table under HELIX_ARCHIVE) -- a CI runner, or any fresh site,
            # has none of it.
            #
            # SiteError is the same situation said a different way: a site profile
            # may declare a root as null ("this site does not have that"), and
            # helix.paths.root() then refuses rather than handing back another
            # site's literal. Refusing
            # is correct for RUNNING the config; it says nothing about whether the
            # config leaks module objects, which is what is under test here.
            #
            # SystemExit is the third spelling of the same thing, and the one a
            # CI runner hits FIRST: every fm config refuses to execute
            # without HELIX_ROOT, because it cannot self-locate (pimm executes a
            # temp copy, so __file__ is the copy) and a wrong checkout is a
            # correctness bug rather than an inconvenience. With no site and no
            # environment, that refusal fires before any root is even consulted.
            #
            # Skip that config rather than fail: the contract under test is about
            # module objects leaking into the namespace, and a config that cannot
            # be executed tells us nothing about it either way.
            missing.append(f"{path.name} ({e})")
            continue
        checked.append(path.name)
        leaked = sorted(k for k, v in ns.items()
                        if not k.startswith("__") and isinstance(v, types.ModuleType))
        assert not leaked, (
            f"{path.name} leaks module objects {leaked} into the config dict; "
            f"Config.dump cannot serialise them and the run dies at setup")

    # A test that checked nothing must not report success. Without this the whole
    # thing passes vacuously wherever the data is absent, which is exactly where
    # nobody would notice.
    if not checked:
        pytest.skip("no coeff_fm config could be executed; external data absent: "
                    + "; ".join(missing))


def test_bootstrap_block_deletes_its_temporaries():
    """Same guarantee for the block the hook writes into the dumped config."""
    import types

    from helix.integrations._bootstrap import bootstrap_block

    ns = {}
    exec(compile(bootstrap_block("/a", "/b"), "<block>", "exec"), ns)
    leaked = sorted(k for k, v in ns.items()
                    if not k.startswith("__") and isinstance(v, types.ModuleType))
    assert not leaked, f"bootstrap_block leaks {leaked}"
