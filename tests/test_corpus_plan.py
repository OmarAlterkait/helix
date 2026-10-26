"""A corpus against its plan (helix.data.coeff_verify.verify_plan) and shard comparison."""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("torch")
from helix.build import build                                  # noqa: E402
from helix.data import coeff_verify                             # noqa: E402
from _build_fixture import PlanesSource                         # noqa: E402
from test_corpus_acceptance import _chain, _plane_fn           # noqa: E402


def _corpus(root, runs=("runA", "runB"), n=4, eps=2):
    """Two runs of ``n`` events, ``eps`` per shard, one global norm table, and their plan."""
    os.makedirs(os.path.join(root, "_calib"))
    lines, chain = [], _chain()
    for i, run in enumerate(runs):
        prefix = os.path.join(root, run, "cx")
        src = PlanesSource([(f"src_{run}.h5", e) for e in range(n * i, n * (i + 1))], _plane_fn, run=run)
        if i == 0:
            cal = os.path.join(root, "_calib", "norm_sigma_global.npy")
            build(src, prefix, indices=list(range(n)), chain=chain, calibrate=cal)
        build(src, prefix, indices=list(range(n)), chain=chain, events_per_shard=eps,
              norm_sigma=np.load(cal))
        lines.append(f"{prefix} 0 {n} {eps} 0 src_{run}.h5")
    open(os.path.join(root, "_calib", "CHUNKS.txt"), "w").write("\n".join(lines) + "\n")
    return root


def _fatal(probs):
    """What a test build cannot avoid: provenance notes, and the tree the test runs from being dirty."""
    return [p for p in probs if "NOTE:" not in p and "DIRTY working tree" not in p]


def test_a_complete_corpus_passes(tmp_path):
    assert _fatal(coeff_verify.verify_plan(_corpus(str(tmp_path)))) == []


def test_a_missing_shard_and_a_stray_one_are_reported(tmp_path):
    root = _corpus(str(tmp_path))
    os.remove(os.path.join(root, "runB", "cx_coeff_0001.h5"))
    os.rename(os.path.join(root, "runA", "cx_coeff_0001.h5"), os.path.join(root, "runA", "cx_coeff_0007.h5"))
    probs = " | ".join(coeff_verify.verify_plan(root))
    assert "runB/cx: shard 1 missing" in probs
    assert "runA/cx: shard 7 is not in the plan" in probs


def test_a_shard_with_its_own_norm_table_is_reported(tmp_path):
    root = _corpus(str(tmp_path))
    np.save(os.path.join(root, "_calib", "norm_sigma_global.npy"),
            np.load(os.path.join(root, "_calib", "norm_sigma_global.npy")) * 2)
    assert any("norm_sigma is not" in p for p in coeff_verify.verify_plan(root))


def test_compare_sees_identical_rebuilds_and_a_changed_value(tmp_path):
    a, b = _corpus(str(tmp_path / "a")), _corpus(str(tmp_path / "b"))
    pa, pb = (os.path.join(r, "runA", "cx_coeff_0000.h5") for r in (a, b))
    assert coeff_verify.compare(pa, pb) == []
    import h5py
    with h5py.File(pb, "r+") as f:
        f["value"][0] += 1
    assert coeff_verify.compare(pa, pb) == ["value: data differ"]
