"""Shared test fixtures: synthetic TPC data generation."""

import importlib.util
import os
import sys

import numpy as np
import pytest

from helix.tpc.config import DetectorConfig


@pytest.fixture
def config():
    return DetectorConfig(
        group_size=64,
        beta=0.15,
        num_time_steps=512,
    )


@pytest.fixture
def synthetic_plane(config):
    """Generate a synthetic plane with known signal, intrinsic noise, and coherent noise."""
    rng = np.random.default_rng(42)
    nw, nt = 256, config.num_time_steps
    gs = config.group_size
    ng = nw // gs

    clean = np.zeros((nw, nt), dtype=np.float32)
    for w in range(80, 140):
        t_center = 200 + (w - 80) * 2
        for dt in range(-15, 16):
            t = t_center + dt
            if 0 <= t < nt:
                clean[w, t] = 40.0 * np.exp(-0.5 * (dt / 5.0) ** 2)

    sigma_w = np.full(nw, 1.5, dtype=np.float32)
    intrinsic = (sigma_w[:, None] * rng.standard_normal((nw, nt))).astype(np.float32)

    coh_waveforms = rng.standard_normal((ng, nt)).astype(np.float32) * 2.5
    coh = np.zeros_like(clean)
    for w in range(nw):
        coh[w] = coh_waveforms[w // gs]

    dig = (clean + intrinsic + coh).astype(np.float32)

    return {
        "clean": clean,
        "dig": dig,
        "coh": coh,
        "sigma_w": sigma_w,
        "nw": nw,
        "nt": nt,
    }


# Pin torch's intra-op threads for the whole suite.
#
# Two reasons, both measured. Speed: these tests run many tiny models, where 20
# threads cost far more in coordination than they save — 381s -> 66s across
# test_model_fm.py + test_training_parity.py. Determinism: CPU float reductions
# split across threads, so the sum order (and the last bits) depend on the thread
# count, which depends on machine load. That is what made the goldens fail
# intermittently before they pinned it themselves.
try:
    import torch
    torch.set_num_threads(4)
except ImportError:
    pass


def pytest_configure(config):
    """Refuse to pretend the pimm seam was checked when pimm is absent.

    Seven test modules gate themselves on `_pimm_importable()` and skip when it
    is False. That is correct for a DSP-only environment -- but the skip is
    silent, and it hid FORTY tests, including every test of the pimm-facing
    evaluator, launcher and WeightEMA code. They had never run: the containers
    that carry the DSP dependencies do not have pimm installed, and every
    invocation used a PYTHONPATH without a pimm checkout on it, so the guard
    returned False every time and the suite reported a clean 287 passed.

    `HELIX_REQUIRE_PIMM=1` turns that into a hard error at startup. Use it for
    any run whose result is meant to mean "the integration is good", and put a
    pimm checkout on PYTHONPATH:

        PYTHONPATH=<pimm-checkout>:<helix> HELIX_REQUIRE_PIMM=1 pytest

    See TESTING.md for the full incantation per container.
    """
    # Site environment BEFORE anything opens a file. HDF5_USE_FILE_LOCKING=FALSE
    # is the one that matters: GPFS does not support the locks h5py takes by
    # default, so every real-shard test dies with "Errno 524 ... unable to lock
    # file" without it. Those tests used to SKIP (their paths did not resolve),
    # so the suite was green and had never opened a real shard here.
    from helix import paths as _paths
    _paths.apply_site_env()

    _require_data()

    # `-q` suppresses pytest_report_header, and that header is the ONE thing
    # that makes a green run quotable: it names the site and says which data
    # roots resolved. The documented command in CLAUDE.md is `pytest tests -q`,
    # so the default invocation was hiding exactly the line that distinguishes
    # "568 passed against the real corpus" from "568 passed against nothing".
    # Print it ourselves when pytest will not.
    # Written to stderr rather than through the terminal reporter: at
    # pytest_configure time the reporter has not begun its session output, and
    # anything handed to it there is swallowed. stderr is not.
    if config.get_verbosity() < 0:
        for line in pytest_report_header(config):
            print(line, file=sys.stderr)

    if os.environ.get("HELIX_REQUIRE_PIMM") != "1":
        return
    try:
        if importlib.util.find_spec("pimm") is None:
            raise ImportError("no pimm on sys.path")
        import pimm.datasets.builder  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise pytest.UsageError(
            f"HELIX_REQUIRE_PIMM=1 but pimm is not importable ({exc}). "
            f"The pimm-facing tests would have SKIPPED silently. Put a pimm "
            f"checkout on PYTHONPATH, or unset HELIX_REQUIRE_PIMM to accept a "
            f"DSP-only run."
        ) from exc


# ---------------------------------------------------------------------------
# jaxtpc fixtures for the dense-chain tests
#
# tests/test_dense_chain.py exercises Densify (pimm-data) -> AddNoise/Digitize
# (helix) end to end, so it needs a jaxtpc sample. These are SYNTHETIC --
# pimm_data.testing builds a minimal cross-modality-consistent v3 dataset from
# numpy + h5py -- so this is a code dependency helix already has (helix.data
# imports pimm_data), not a data dependency on the simulator.
#
# The *_DATA_ROOT env vars point at real shards when set, matching pimm-data's
# own conftest so the same command means the same thing in both repos.
# ---------------------------------------------------------------------------

def _jaxtpc_root(env, factory, maker, name):
    import os
    real = os.environ.get(env)
    if real:
        return real
    root = factory.mktemp(name)
    maker(str(root))
    return str(root)


@pytest.fixture(scope="session")
def jaxtpc_data_root(tmp_path_factory):
    testing = pytest.importorskip("pimm_data.testing")
    return _jaxtpc_root("JAXTPC_DATA_ROOT", tmp_path_factory,
                        testing.make_jaxtpc_sample, "jaxtpc_synth")


@pytest.fixture(scope="session")
def jaxtpc_pixel_data_root(tmp_path_factory):
    testing = pytest.importorskip("pimm_data.testing")
    maker = getattr(testing, "make_jaxtpc_pixel_sample", None)
    if maker is None:
        pytest.skip("pimm_data.testing has no pixel sample builder")
    return _jaxtpc_root("JAXTPC_PIXEL_DATA_ROOT", tmp_path_factory,
                        maker, "jaxtpc_pixel_synth")


def pytest_report_header(config):
    """Say which site is selected and which data roots resolved.

    A skip because data is absent looks exactly like a skip because the machine
    has no GPU, and the suite prints neither. Putting the site and the roots in
    the header means a run whose result you are about to quote states what it
    was actually able to reach.
    """
    from helix import paths
    # `import _paths`, not `import tests._paths`. helix's tests/ has no
    # __init__.py, so `tests` is only ever a NAMESPACE package -- and a
    # namespace package loses to any regular `tests` package on sys.path.
    # The pimm checkout ships one, and helix_run.sh puts that checkout on
    # PYTHONPATH, so inside the container this line resolved to pimm's
    # tests and died with ModuleNotFoundError before a single test ran.
    # Every test module in this directory already says `from _paths import
    # ...`; these two lines were the only ones that did not.
    import _paths as tp

    site = paths.site_name() or "NONE (nothing selected or detected)"
    present = [n for n, v in (("corpus", tp.CORPUS), ("sensor", tp.SENSOR_ROOT),
                              ("pimm", tp.PIMM_ROOT)) if v and os.path.isdir(v)]
    absent = [n for n, v in (("corpus", tp.CORPUS), ("sensor", tp.SENSOR_ROOT),
                             ("pimm", tp.PIMM_ROOT)) if not (v and os.path.isdir(v))]
    return [f"helix site: {site}",
            f"helix data: present={','.join(present) or '-'}  "
            f"absent={','.join(absent) or '-'}"]


def _require_data():
    """`HELIX_REQUIRE_DATA=1` -> a missing data root is an ERROR, not a skip.

    The counterpart to HELIX_REQUIRE_PIMM, and it exists for the same measured
    reason one level along. Roughly ten test modules gate on
    `os.path.isdir(CORPUS)` or `os.path.exists(<shard>)` and skip when the path
    is wrong -- including every test that reads the real corpus, the bin-grid
    fingerprint check that pins the handover claim, and the corpus-identity
    guard. A mistyped HELIX_CORPUS, or a site profile naming a directory that
    was never copied, therefore produces a green run that checked none of it.

    Use it for any run whose result is meant to mean "this corpus is good".
    """
    if os.environ.get("HELIX_REQUIRE_DATA") != "1":
        return
    import _paths as tp          # see pytest_report_header for why not tests._paths
    from helix import paths

    missing = {n: v for n, v in (("HELIX_CORPUS", tp.CORPUS),
                                 ("HELIX_SENSOR_ROOT", tp.SENSOR_ROOT))
               if not (v and os.path.isdir(v))}
    if missing:
        detail = "; ".join(f"{k}={v or '(unset)'}" for k, v in missing.items())
        raise pytest.UsageError(
            f"HELIX_REQUIRE_DATA=1 but these roots do not resolve to a directory: "
            f"{detail}. site={paths.site_name() or 'none'}. The real-data tests "
            f"would have SKIPPED silently. Run `python -m helix.paths`, or unset "
            f"HELIX_REQUIRE_DATA to accept a no-data run."
        )
