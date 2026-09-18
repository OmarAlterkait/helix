"""Every entry point of helix/data/transforms.py actually executes.

This file exists because the module shipped BROKEN and both suites stayed green.
helix/data/transforms.py was moved out of pimm_data verbatim, carrying relative
imports that were valid there and are not here -- ``from . import dense_ops``,
``from .batch_transforms import ...``, ``from .transform import Compose`` -- and
losing its module-level ``import hashlib``. Four of its five code paths raised
ImportError or NameError on first call.

Nothing caught it because nothing IMPORTED it. tests/test_boundary.py probed the
PACKAGE ``helix.data``, whose ``__init__`` pulls only coeff_reader and
coeff_dataset; the registered transforms are reached by string through the
registry, so no test ever named the module. A module that is only ever addressed
by string needs a test that addresses it by import.

These are deliberately shallow: they assert that each path RUNS, not what it
computes. The numerical behaviour is pinned by tests/test_forward_noise.py.
"""

import numpy as np
import pytest

pytest.importorskip("pimm_data")
pytest.importorskip("torch")

import helix.data.transforms as T  # noqa: E402
from pimm_data.transform import TRANSFORMS, Compose  # noqa: E402

GEOM = {0: dict(label="volume_0_U", n_wires=8, n_ticks=64, pedestal=0,
                wire_lengths=np.full(8, 2.33, np.float32))}


def test_module_has_no_relative_imports():
    """`from .x import y` inside helix.data resolves to helix.data.x.

    The module was copied out of pimm_data, where those spellings were correct.
    Three of them survived the move and named modules helix.data does not have.
    """
    import re
    src = open(T.__file__, encoding="utf-8").read()
    bad = re.findall(r"^\s*from \.\S*\s+import.*$", src, re.M)
    assert not bad, f"relative imports left over from pimm_data: {bad}"


def test_every_name_the_module_uses_is_imported():
    """`hashlib` was used by _event_rng and not imported -- NameError on call."""
    a = T.AddNoise(geom=GEOM, modality="sensor", coherent=True, incoherent=True,
                   wire_lengths_m=2.33)
    r = a._event_rng("ev0")
    assert r.standard_normal(1).shape == (1,)


def test_event_rng_is_deterministic_per_name():
    a = T.AddNoise(geom=GEOM, modality="sensor", coherent=True, incoherent=True,
                   wire_lengths_m=2.33)
    first = a._event_rng("ev0").standard_normal(4)
    assert np.array_equal(first, a._event_rng("ev0").standard_normal(4))
    assert not np.array_equal(first, a._event_rng("ev1").standard_normal(4))


@pytest.mark.parametrize("name", ["AddNoise", "Digitize"])
def test_importing_the_module_registers_the_transform(name):
    """The builder names these by STRING, so registration is the whole contract.

    scripts/build_coeff_corpus.py composes them from dicts. Before the move,
    `from pimm_data.transform import Compose` registered them transitively; it
    no longer does, and the builder raised "AddNoise is not in the transforms
    registry" -- at which point the production corpus build could not run.
    """
    assert TRANSFORMS.get(name) is not None


def test_compose_can_build_them_by_name():
    Compose([dict(type="Digitize", geom=GEOM, modality="sensor", n_bits=12)])


def test_recipe_builders_run():
    """sensor_dense_cfg / build_sensor_gpu_stages -- the latter reached
    `from .transform import Compose` and died with ModuleNotFoundError."""
    cfg = T.sensor_dense_cfg(GEOM, modality="sensor")
    assert isinstance(cfg, list) and cfg, "recipe must not be empty"
    stages = T.build_sensor_gpu_stages(GEOM, modality="sensor")
    assert stages is not None


def test_the_builder_registers_before_it_composes():
    """scripts/build_coeff_corpus.py must import this module, not rely on
    pimm_data pulling it in. Checked as source, because running the builder
    needs real shards and a GPU.

    Resolved through helix.paths.repo() rather than opened by relative path: the
    test read "scripts/build_coeff_corpus.py" from the process CWD, so it passed
    only when pytest happened to be invoked from the repo root and raised
    FileNotFoundError from anywhere else. It was the one relative open() in the
    suite."""
    from helix.paths import repo

    src = (repo() / "scripts" / "build_coeff_corpus.py").read_text(encoding="utf-8")
    i_import = src.find("import helix.data.transforms")
    i_use = src.find('type="AddNoise"')
    assert i_import != -1, "builder composes AddNoise but never registers it"
    assert i_use == -1 or i_import < i_use, "registration must precede the Compose"


def test_addnoise_defaults_come_from_the_noise_module():
    """Not literals repeating it.

    The signature imported five constants from helix.tpc.noise and then re-typed
    their values inline. They agreed, so nothing failed -- but retuning the
    forward model would have left the registered transform on the old values,
    with the corpus builder and the trainer disagreeing silently. Same drift as
    DetectorConfig's third copy of the ENC triple (test_forward_noise.py).
    """
    import inspect
    from helix.tpc import noise as N
    d = {k: v.default for k, v in
         inspect.signature(T.AddNoise.__init__).parameters.items()}
    assert d["group_size"] == N.DEFAULT_GROUP_SIZE
    assert d["coh_rms"] == N.DEFAULT_COH_RMS_ADC
    assert d["coh_corner_freq_hz"] == N.DEFAULT_COH_CORNER_FREQ_HZ
    assert d["coh_spectral_slope"] == N.DEFAULT_COH_SLOPE
    assert d["beta"] == N.DEFAULT_COH_BETA
    assert d["enc"] == N.DEFAULT_ENC
