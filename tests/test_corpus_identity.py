"""A checkpoint and a corpus can be mismatched with nothing to say so.

Each corpus is internally consistent -- CoeffTPCReader rejects shards that
disagree with EACH OTHER -- so a reader pointed at the wrong corpus is perfectly
satisfied. eval_checkpoint validated only that the split name exists, which every
corpus satisfies. The failure mode was therefore a plausible number, not an
error, and that is the hardest kind to notice.

The live instance: coeff_tpc and coeff_tpc_r1 share a run name and differ only in
the coherent-removal gate (r1 records tau=0.05). Same wavelet, bands, gids,
sigma_norm, noise model; different surviving coefficients.
"""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from helix.data.identity import corpus_identity, check_corpus_matches  # noqa: E402


def _shard(tmp_path, digest, removal, name="sim_wire_coeff_0000.h5"):
    p = tmp_path / name
    with h5py.File(p, "w") as f:
        cfg = f.create_group("config")
        cfg.attrs["basis_digest"] = digest
        cfg.attrs["removal_json"] = removal
        cfg.attrs["sigma_norm"] = 2.6
    return p


R1 = ('8c4542b6', '{"kgate": 3.0, "npass": 2, "tau": 0.05}')
LEGACY = ('7f954a84', '{"kgate": 3.0, "npass": 2}')


def test_identity_is_read_from_the_shard(tmp_path):
    _shard(tmp_path, *R1)
    got = corpus_identity(str(tmp_path))
    assert got["basis_digest"] == R1[0]
    assert "tau" in got["removal_json"]


def test_a_missing_corpus_is_not_a_mismatch(tmp_path):
    """'not a corpus' and 'the wrong corpus' are different failures."""
    assert corpus_identity(str(tmp_path)) is None
    with pytest.raises(ValueError, match="not a corpus"):
        check_corpus_matches({"basis_digest": R1[0]}, None, where="x")


def test_matching_digests_pass(tmp_path):
    _shard(tmp_path, *R1)
    actual = corpus_identity(str(tmp_path))
    assert "OK" in check_corpus_matches(actual, actual, where="x")


def test_the_gate_difference_is_refused(tmp_path):
    """The whole point: two corpora that differ ONLY in tau must not be swapped."""
    _shard(tmp_path, *R1)
    r1 = corpus_identity(str(tmp_path))
    d2 = tmp_path / "legacy"; d2.mkdir()
    _shard(d2, *LEGACY)
    legacy = corpus_identity(str(d2))

    with pytest.raises(ValueError) as e:
        check_corpus_matches(r1, legacy, where="eval")
    msg = str(e.value)
    assert "corpus mismatch" in msg
    # the message must show BOTH gates, or the reader cannot tell WHICH is which
    assert "tau" in msg and "different DSP" in msg


def test_an_unstamped_run_warns_but_does_not_fail(tmp_path):
    """Every checkpoint trained before the stamp existed has no record.

    Refusing those would make the guard unadoptable, so it must degrade to the
    pre-existing behaviour -- unchecked -- and SAY so.
    """
    _shard(tmp_path, *R1)
    actual = corpus_identity(str(tmp_path))
    for recorded in (None, {}, {"basis_digest": ""}):
        note = check_corpus_matches(recorded, actual, where="x")
        assert "NOT RECORDED" in note
