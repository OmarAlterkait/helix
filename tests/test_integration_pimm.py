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
