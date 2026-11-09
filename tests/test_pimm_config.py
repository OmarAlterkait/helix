"""The pimm config contract: helix's configs and the registry adapters against what pimm
reads, and the integration's import boundaries."""
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
    # one rank's batch: scripts/submit.sh appends batch_size=<nodes x GPUs> (docs/ARCHITECTURE.md §8)
    assert cfg.batch_size == 1
    assert cfg.model["type"] == "Coeff-FM"
    assert cfg.data["train"]["type"] == "CoeffTPCDataset"


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


def test_configs_are_world_size_free():
    """The FM needs batch_size_per_gpu == 1: it has no event separation, so a rank holding two
    events would attend across them (docs/ARCHITECTURE.md §8). pimm's ``batch_size`` is GLOBAL;
    the configs hold one rank's value (1) and read no world size at load -- scripts/submit.sh
    appends the launch's batch, and FMTrainer scales the LR -- so a config means the same thing
    on the submitting host as on the job."""
    for path in sorted(pathlib.Path(CONFIG).parent.glob("*.py")):
        src = path.read_text()
        # read at load: the env var or a helper call. A declared ``required_world_size`` (checked by
        # FMTrainer at run time, configs/pimm/ft_eval.py) is a constant, not a read.
        assert "WORLD_SIZE" not in src and "world_size(" not in src, path.name
        tree = ast.parse(src)
        vals = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
        for key in ("batch_size", "batch_size_val", "batch_size_test"):
            v = vals.get(key)
            if path.name == "fm.py":
                assert v is not None, f"fm.py does not set {key}"
            assert v is None or (isinstance(v, ast.Constant) and v.value == 1), \
                f"{path.name}: {key} must be 1 (one rank's batch) or inherited"


def test_submit_appends_the_world_size_as_training_overrides():
    """scripts/submit.sh turns nodes x GPUs into the batch overrides, after `--`, last."""
    src = (pathlib.Path(CONFIG).parent.parent.parent / "scripts" / "submit.sh").read_text()
    assert "W=$((NN * NP))" in src
    for key in ("batch_size=$W", "batch_size_val=$W", "batch_size_test=$W", "num_worker=$((W *"):
        assert key in src, key
    assert 'REST+=(-- "${BATCH[@]}")' in src and 'REST+=("${BATCH[@]}")' in src


@pimm_required
def test_the_peak_lr_scales_as_sqrt_world_from_its_reference():
    """fm.py's LR was ``4.4e-3 * sqrt(world / 16)`` computed in the config; FMTrainer now applies
    the same rule, so the numbers are unchanged at every GPU count (and exact at 16)."""
    from helix.integrations.pimm.trainer import scaled_lr
    assert scaled_lr(4.4e-3, 16, 16) == 4.4e-3
    for world in (1, 4, 8, 32, 64):
        assert scaled_lr(4.4e-3, 16, world) == pytest.approx(4.4e-3 * (world / 16) ** 0.5, rel=1e-15)
    assert scaled_lr(3e-3, None, 4) == 3e-3                        # the denoiser's: absolute


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


def test_no_config_evaluates_during_training():
    """Validation loss does not rank representations, so no config evaluates in training: no
    evaluator hook, ``evaluate`` off in the base (pimm then builds no val/test loader), and the
    saver is CheckpointSaverIteration, which writes ``model/last`` on its cadence and nothing
    else -- pimm's CheckpointSaver also writes ``iter_N.pth`` on every save step."""
    import ast
    names = {}
    for path in sorted(pathlib.Path(CONFIG).parent.glob("*.py")):
        tree = ast.parse(path.read_text())
        names[path.name] = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign)
                            for t in n.targets if isinstance(t, ast.Name)}
        hooks = names[path.name].get("hooks")
        if hooks is None:
            continue
        types = [kw.value.value for h in hooks.elts for kw in h.keywords if kw.arg == "type"]
        assert not any(t.endswith("Evaluator") for t in types), f"{path.name}: {types}"
        assert types[-1] == "CheckpointSaverIteration", f"{path.name}: saver is {types[-1]}"
    assert ast.literal_eval(names["fm.py"]["evaluate"]) is False
    assert all("evaluate" not in v or ast.literal_eval(v["evaluate"]) is False for v in names.values())


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


def test_configs_leave_no_module_objects_in_the_namespace():
    """A config must not leave a module object (an `import os as _os`) in its namespace.

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


def test_the_fm_variants_inherit_the_base_hooks():
    """fm_cooldown.py and fm_smoke.py take fm.py's hook list rather than restating it (a list is
    replaced, not merged, so a restated copy drifts). The cooldown used to restate it to drop
    WeightEMA; with no EMA anywhere there is nothing to drop."""
    for name in ("fm_cooldown.py", "fm_smoke.py"):
        src = (pathlib.Path(CONFIG).parent / name).read_text()
        assert "hooks" not in _config_names(pathlib.Path(CONFIG).parent / name), name
        assert "WeightEMA" not in src
    assert "WeightEMA" not in pathlib.Path(CONFIG).read_text()


@pimm_required
def test_init_warm_starts_from_a_runs_last_and_is_not_reread(tmp_path):
    """``Coeff-FM``'s ``init`` loads an FM run's ``model/last`` into the freshly built model (the
    cooldown's warm start, in place of pimm's weight=), and reading the warm-started run back
    builds its architecture without touching ``init`` -- whose run may be gone by then."""
    torch = pytest.importorskip("torch")
    from test_eval import SMALL, _config, _same, _write_run

    import helix.integrations.pimm  # noqa: F401
    from helix.model.checkpoint import model_config
    from pimm.models.builder import MODELS

    raw, ema = _write_run(tmp_path / "stable")
    bins = dict(edges=raw.bin_edges.clone())
    m = MODELS.build(dict(type="Coeff-FM", bins=bins, init=str(tmp_path / "stable"), **SMALL))
    assert _same(m, raw) and not _same(m, ema)
    cool = _config()
    cool["model"]["init"] = str(tmp_path / "gone")
    assert "init" not in model_config(cool)
    assert model_config(cool) == model_config(_config())


def test_training_runs_a_pinned_pimm():
    """A site that pins pimm names a full commit, and scripts/submit.sh refuses any other HEAD or a
    dirty checkout at HELIX_PIMM_ROOT. Where that checkout is reachable, it IS at the pin."""
    import re
    import subprocess
    import yaml
    root = pathlib.Path(CONFIG).parent.parent.parent
    src = (root / "scripts" / "submit.sh").read_text()
    assert "HELIX_SLURM_TRAIN_PIMM_REV" in src and "rev-parse HEAD" in src and "status --porcelain" in src
    site = yaml.safe_load((root / "helix" / "sites" / "nersc.yaml").read_text())
    rev = site["scheduler"]["train"]["pimm_rev"]
    assert re.fullmatch(r"[0-9a-f]{40}", rev)
    pimm = os.path.expandvars(site["roots"]["HELIX_PIMM_ROOT"]) if "roots" in site else None
    import shutil
    if not pimm or not os.path.isdir(pimm) or shutil.which("git") is None:
        pytest.skip("the pinned pimm checkout is not reachable here")
    head = subprocess.run(["git", "-C", pimm, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == rev, f"{pimm} is at {head}, the NERSC profile pins {rev}"


def test_the_pimm_checkout_pins_helixs_pimm_data():
    """The forward-model registration is LOCKSTEP (pimm_data/_registry.py raises KeyError on a
    duplicate), so the pimm that trains must pin the pimm-data revision helix pins in pyproject.toml
    (which container/install.sh reads; CI's ``uv lock --check`` holds the lockfile to it). This is
    the copy in another repository, checked where that checkout is reachable."""
    import re
    pat = r'pimm-data\s*=\s*\{[^}]*rev\s*=\s*"([0-9a-f]{40})"'
    ours = re.search(pat, (pathlib.Path(CONFIG).parent.parent.parent / "pyproject.toml").read_text())
    assert ours, "pyproject.toml no longer pins pimm-data by an explicit 40-char rev"
    p = pathlib.Path(PIMM) / "pyproject.toml" if PIMM else None
    if p is None or not p.exists():
        pytest.skip("no pimm checkout reachable (HELIX_PIMM_ROOT)")
    theirs = re.search(pat, p.read_text())
    if theirs is None:
        pytest.skip(f"{p} does not pin pimm-data by an explicit rev")
    assert theirs.group(1) == ours.group(1), f"{p} pins pimm-data {theirs.group(1)[:8]}, helix {ours.group(1)[:8]}"
