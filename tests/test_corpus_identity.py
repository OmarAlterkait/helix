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
# identity.py itself imports no pimm_data -- it is pure h5py -- but
# helix/data/__init__.py eagerly imports coeff_reader and coeff_dataset, which
# do. So importing any helix.data submodule requires the optional [pimm] extra,
# and without this guard the module raised at collection and failed the suite.
pytest.importorskip("pimm_data")

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


# ── the resume path, not just eval ───────────────────────────────────────────

def test_resuming_against_a_different_corpus_is_refused():
    """A run directory continued against a different corpus must REFUSE.

    check_corpus_matches was wired into scripts/eval_checkpoint.py alone. The
    training hook RECORDED corpus identity per link and never compared it, so a
    resumed run could switch corpora silently -- which is the exact mismatch
    identity.py exists to prevent, in the one place where it also costs GPU
    days before anyone sees a number.
    """
    r1 = dict(basis_digest="8c4542b6" + "0" * 56, removal_json='{"tau":0.05}',
              sigma_norm=2.6)
    legacy = dict(basis_digest="7f954a84" + "0" * 56, removal_json="{}",
                  sigma_norm=2.6)
    with pytest.raises(Exception) as e:
        check_corpus_matches(r1, legacy, where="run/provenance.json")
    assert "basis_digest" in str(e.value) or "corpus" in str(e.value).lower()


def test_the_training_hook_actually_calls_the_guard():
    """Guard the wiring, not just the guard.

    The function existing and being correct is not the property that matters
    here; being CALLED from the resume path is.
    """
    # Read the FILE, do not import it: helix.integrations.pimm imports the pimm
    # framework at module scope (deliberately -- see helix/integrations/__init__),
    # and the DSP container has no pimm. A test of the wiring must not require
    # the very environment the wiring is for.
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "helix", "integrations", "pimm", "hooks.py"),
               encoding="utf-8").read()
    # A SUBSTRING TEST IS NOT ENOUGH. The first version of this asserted
    # `"check_corpus_matches" in src`, and a mutation that deleted the import
    # AND the call still passed -- because the name also appears in the comment
    # explaining why the call is there. Parse for an actual Call node.
    import ast
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", None))
             == "check_corpus_matches"]
    assert calls, (
        "the training hook no longer CALLS check_corpus_matches (a comment "
        "mentioning it does not count) -- a resumed run can switch corpora "
        "silently again")
