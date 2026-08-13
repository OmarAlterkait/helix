"""The pimm adapter must register helix's names — without pimm knowing helix.

``helix/integrations/pimm.py`` is how a pimm config reaches the coefficient FM.
It lives here rather than inside pimm-private so the dependency points one way:
helix knows how to plug into pimm, pimm knows nothing about helix. A pimm config
pulls it in with mmcv's standard ``custom_imports`` hook, which pimm's
``Config.fromfile`` already honours.

These tests skip when pimm is not importable — which includes environments that
have pimm checked out but not its dependencies.
"""

import importlib.util
import os
from pathlib import Path

import pytest


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
                 "audit_shard", "coord_digest", "build_corpus_stream",
                 "DetectorConfig", "process_event", "get_backend"):
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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = Config.fromfile(os.path.join(root, "configs", "pimm", "coeff_fm_encode.py"))
    assert "helix.integrations.pimm" in cfg.custom_imports["imports"]
    assert cfg.batch_size == 1, "the FM has no event separation; see MULTI_EVENT_BATCHING.md"
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
    from pimm_data.coeff import CoeffTPCDataset as Inner

    src = (Path(__file__).resolve().parent.parent
           / "helix" / "integrations" / "pimm.py").read_text()
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
