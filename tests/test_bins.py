"""The bin grid is reproducible, and a mismatch is reported as a mismatch.

The grid is the objective the categorical head trains against; it is derived from corpus
statistics (the FIRST ``events`` events, no sampling) and a table records its parameters, so the
corpus reproduces it exactly and the 8 KB file need not travel. Pinned: ``compare`` detects a
difference; rederiving the production grid from a real corpus is bit-identical; the packaged
fingerprint ships and rejects a wrong grid (all a site that arrived without the file can check).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

pytest.importorskip("pimm_data")          # the derivation reads the corpus through CoeffTPCDataset

from helix.data import bins as binlib                            # noqa: E402

from _fixtures import CORPUS                                        # noqa: E402


def _table(seed=0, K=8, n_bands=2, corpus="/somewhere/run_1", events=120):
    """A structurally valid table. Values are arbitrary; shapes are not."""
    rng = np.random.default_rng(seed)
    return dict(
        edges=rng.normal(size=(n_bands, K + 1)).astype(np.float32),
        cent_asinh=rng.normal(size=(n_bands, K)).astype(np.float32),
        cent_ratio=rng.normal(size=(n_bands, K)).astype(np.float32),
        K=K, n_bands=n_bands, corpus=corpus, events=events)


def test_compare_detects_every_kind_of_difference():
    """A verifier that always passes is worse than none: one moved edge is a different objective,
    another K is not the same grid, and a wrong-shaped table is reported, not raised."""
    a = _table()
    assert binlib.compare(a, a)["identical"]
    b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
    b["edges"][0, 0] += np.float32(1e-3)
    r = binlib.compare(a, b)
    assert not r["identical"] and r["arrays"]["edges"]["max_abs_diff"] > 0 and r["arrays"]["cent_asinh"]["identical"]
    assert binlib.compare(a, dict(a, K=16))["param_mismatches"]["K"] == (8, 16)
    r = binlib.compare(_table(K=8), _table(K=16))
    assert not r["identical"] and r["arrays"]["edges"]["max_abs_diff"] == float("inf")


def test_params_of_fills_what_a_table_does_not_record_from_defaults():
    """A table records corpus/events/K/n_bands, never the percentiles; an older sidecar missing
    keys still rederives."""
    p = binlib.params_of(_table(K=8, n_bands=2, events=99, corpus="/c/run_2"))
    assert (p["K"], p["n_bands"], p["events"], p["corpus"]) == (8, 2, 99, "/c/run_2")
    assert all(p[k] == binlib.DEFAULTS[k] for k in ("lo_pct", "hi_pct", "dataset_name"))
    p = binlib.params_of(dict(edges=None, corpus="/c/run_3"))
    assert p["corpus"] == "/c/run_3" and p["K"] == binlib.DEFAULTS["K"] and p["events"] == binlib.DEFAULTS["events"]


def test_save_load_roundtrip_is_exact(tmp_path):
    """float32 in, float32 out -- a lossy roundtrip would fail --verify forever."""
    pytest.importorskip("torch")
    t = _table()
    p = tmp_path / "bins.pt"
    binlib.save(t, p)
    back = binlib.load(p)

    assert binlib.compare(t, back)["identical"]
    for k in binlib.ARRAY_KEYS:
        assert back[k].dtype == np.float32
    assert (back["K"], back["n_bands"], back["corpus"]) == (t["K"], t["n_bands"], t["corpus"])


@pytest.mark.skipif(not os.path.isdir(CORPUS), reason=f"corpus absent: {CORPUS}")
def test_rederiving_the_production_grid_reproduces_it_bit_identically():
    """The handover claim, against the real corpus.

    Slow -- it pools 120 events. It is the test that licenses NOT copying the
    bin table to a receiving site, so it earns its runtime.
    """
    pytest.importorskip("torch")

    # From reference_bins.json's declared name, not a literal. This said `_v2`
    # while the JSON said `_v3`, and because the guard below SKIPS on absence the
    # drift was silent -- the test that exists to check the handover claim simply
    # stopped running.
    ref_path = str(binlib.reference_table())
    if not os.path.exists(ref_path):
        pytest.skip(f"reference grid absent: {ref_path}")

    ref = binlib.load(ref_path)
    got = binlib.rederive(ref, corpus=CORPUS)

    r = binlib.compare(ref, got)
    assert r["identical"], r


# --- the packaged reference: the only check a site WITHOUT the original has ----

def test_the_reference_fingerprint_ships():
    """If this file is missing from an install, check_reference degrades to
    "no-reference" and a wrong-run grid passes unnoticed. pyproject declares it
    under [tool.setuptools.package-data]; this asserts it is actually there."""
    ref = binlib.reference()
    assert ref is not None, f"no packaged reference at {binlib.reference_path()}"
    assert ref["corpus"]["run"] == "j000000"
    assert set(ref["digests"]) == set(binlib.ARRAY_KEYS)
    assert ref["params"]["K"] == 128 and ref["params"]["n_bands"] == 4
    assert ref["params"]["events"] == 120


def test_the_spec_is_the_single_source_of_the_parameters():
    """DEFAULTS must come FROM the spec, not sit beside it as a second copy.

    The parameters are configuration; two places that can disagree is the bug
    this arrangement exists to prevent."""
    ref = binlib.reference()
    for k, v in ref["params"].items():
        assert binlib.DEFAULTS[k] == v, f"{k}: DEFAULTS says {binlib.DEFAULTS[k]}, spec says {v}"


def test_fingerprint_is_stable_and_shape_sensitive():
    t = _table(K=8, n_bands=2)
    assert binlib.fingerprint(t) == binlib.fingerprint(t)
    assert all(v.startswith("sha256:") for v in binlib.fingerprint(t).values())
    assert binlib.fingerprint(t) != binlib.fingerprint(_table(seed=1, K=8, n_bands=2))


def test_check_reference_reports_params_when_the_grid_is_a_different_shape():
    """A K=8 toy is not comparable to the K=128 reference, and saying
    "mismatch" there would be wrong -- nothing was reproduced incorrectly."""
    r = binlib.check_reference(_table(K=8, n_bands=2))
    assert r["status"] == "params"
    assert "K" in r["param_mismatches"]


def test_check_reference_flags_a_same_shape_grid_that_is_not_the_reference():
    """The wrong-run case: right parameters, wrong numbers. This is the failure
    a site with no original grid cannot otherwise detect."""
    ref = binlib.reference()
    K, nb = ref["params"]["K"], ref["params"]["n_bands"]
    r = binlib.check_reference(_table(seed=7, K=K, n_bands=nb,
                                      events=ref["params"]["events"]))
    assert r["status"] == "mismatch"
    assert set(r["digest_mismatches"]) == set(binlib.ARRAY_KEYS)


@pytest.mark.skipif(not os.path.isdir(CORPUS), reason=f"corpus absent: {CORPUS}")
def test_the_production_grid_matches_its_own_packaged_fingerprint():
    """Closes the loop: rederive from the corpus, and the packaged fingerprint
    -- with no reference .pt anywhere -- confirms it is the released grid."""
    pytest.importorskip("torch")
    # No parameters passed: DEFAULTS comes from the spec, so a bare derive is
    # BY CONSTRUCTION the declared one. Passing them here would test the test's
    # copy of the config rather than the one the CLI actually uses. In a child
    # process with the canonical numpy features: this process's numpy may already
    # dispatch to AVX-512 kernels, whose last-bit rounding changes the digests.
    import json, subprocess, sys
    code = ("import json; from helix.data import bins as b; r = b.check_reference(b.derive(%r)); "
            "print(json.dumps(dict(status=r['status'], avx512=b.avx512_active())))" % str(CORPUS))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, **binlib.CANONICAL_NUMPY_ENV))
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert r == dict(status="match", avx512=False), r


@pytest.mark.skipif(not os.path.isdir(CORPUS), reason=f"corpus absent: {CORPUS}")
def test_check_corpus_accepts_the_declared_corpus():
    """The cheap input-side check, against the real thing."""
    assert binlib.check_corpus(CORPUS)["status"] == "ok"
