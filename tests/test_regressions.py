"""Guards for defects that produced a wrong number without failing.

Each test here corresponds to a bug that ran in production and reported
nothing. They are grouped because they share that property, not because they
share a subject: a silent wrong answer needs a test more than a loud one does.
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# MAD sigma: np.partition(a, mid) does not order position mid-1
# --------------------------------------------------------------------------

def test_mad_sigma_numpy_matches_true_median():
    """The even-length branch must be the real median of |residual|.

    ``np.partition(a, mid)`` guarantees only that position ``mid`` holds the
    mid-th order statistic; everything below it is unordered. Reading
    ``p[:, mid-1]`` therefore returned an arbitrary element below the median.
    Tick counts are even in production, so this was the branch that always ran.
    """
    from helix.tpc.coherent_ops_numpy import mad_sigma_per_wire

    rng = np.random.default_rng(0)
    for n_t in (256, 4096):                       # even: the production case
        res = rng.normal(0, 1.0, size=(500, n_t))
        want = np.median(np.abs(res), axis=1) / 0.6745
        np.testing.assert_allclose(mad_sigma_per_wire(res), want,
                                   rtol=1e-6, atol=0)
    for n_t in (257, 4095):                       # odd: was already correct
        res = rng.normal(0, 1.0, size=(500, n_t))
        want = np.median(np.abs(res), axis=1) / 0.6745
        np.testing.assert_allclose(mad_sigma_per_wire(res), want,
                                   rtol=1e-6, atol=0)


def test_mad_sigma_backends_agree():
    """numpy and torch are hand-written twins; nothing else diffs them.

    The whole justification for keeping a separate implementation per backend is
    that a reference exists to compare against. That only holds if something
    actually compares. The partition bug put these two out of step and no test
    noticed.
    """
    torch = pytest.importorskip("torch")
    from helix.tpc.coherent_ops_numpy import mad_sigma_per_wire as np_mad
    from helix.tpc.coherent_ops_torch import mad_sigma_per_wire as t_mad

    rng = np.random.default_rng(1)
    res = rng.normal(0, 1.0, size=(500, 4096)).astype(np.float32)
    np.testing.assert_allclose(np_mad(res),
                               t_mad(torch.from_numpy(res)).numpy(),
                               rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# Event-name seeding: hash() is salted per process
# --------------------------------------------------------------------------

def test_name_hash_is_stable_across_processes():
    """The tokenizer's per-event seed must be a function of the event.

    It was ``hash(name) & 0xFFFFFFFF``. Python salts string hashing per
    interpreter, so a resumed run, a re-run and each DDP rank drew different
    masks for the same event while the code read as though the seed were
    deterministic. Pinned against an independent digest so a future rewrite
    cannot quietly reintroduce process-dependence.
    """
    from helix.model.tokenize import _name_hash

    for name in ("", "event_0", "run_0027575715/event_41", "événement"):
        want = int.from_bytes(
            hashlib.blake2b(name.encode(), digest_size=4).digest(), "big")
        assert _name_hash(name) == want
        assert 0 <= _name_hash(name) < 2 ** 32

    # Distinct events must not collide onto one seed.
    names = [f"event_{i}" for i in range(2000)]
    assert len({_name_hash(n) for n in names}) == len(names)


def test_no_builtin_hash_used_for_seeding():
    """Catch the pattern, not just this one call site.

    Parsed rather than grepped: a text scan also matches the docstring that
    quotes the old expression, which is documentation worth keeping.
    """
    import ast

    src = (REPO / "helix" / "model" / "tokenize.py").read_text()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "hash"]
    assert not calls, (
        f"builtin hash() called at line(s) {[n.lineno for n in calls]} — it is "
        f"salted per process; use _name_hash() for anything feeding an RNG seed")


# --------------------------------------------------------------------------
# Version: stamped into every shard, and checked across shards by pimm-data
# --------------------------------------------------------------------------

def test_version_is_single_sourced():
    """pyproject must not carry its own literal.

    ``helix/core/coeff_io.py`` stamps ``helix.__version__`` into every corpus
    shard and ``pimm_data/coeff_verify.py`` raises on a mismatch between shards.
    When pyproject said 0.1.0 and the module said 0.2.0, that provenance chain
    reported a version the artifact was not built with.
    """
    import helix

    text = (REPO / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text
    assert re.search(r'version\s*=\s*\{\s*attr\s*=\s*"helix\.__version__"\s*\}',
                     text), "pyproject must read the version from the module"
    assert not re.search(r'^version\s*=\s*"', text, re.M), (
        "a literal version in [project] shadows the dynamic one")
    assert re.fullmatch(r"\d+\.\d+\.\d+", helix.__version__)
