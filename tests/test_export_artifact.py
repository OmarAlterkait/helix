"""Promoting a `pimm export` into something that can attribute its own number.

The three things an export structurally cannot record — which weight set, which
corpus, which code — and the refusals that stop an unattributable artifact from
being written at all.
"""
import importlib.util
import json
import os
import pathlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from helix.model.artifact import inspect
# Top-level, not `from tests.…`: there is no tests/__init__.py, so `tests` is
# a namespace package that resolves only when the REPO ROOT happens to be on
# sys.path. That depends on how pytest was invoked -- it held in the
# development tree and broke in a clean clone. pytest's prepend import mode
# puts tests/ itself on sys.path, which is what the rest of this suite relies
# on (see `from _paths import ...`).
from test_artifact_formats import ARCH, OP, make_pimm_export


def _script():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "export_artifact.py"
    spec = importlib.util.spec_from_file_location("_export_artifact_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_promotion_records_what_the_export_could_not(tmp_path):
    src = make_pimm_export(str(tmp_path / "exp"))
    out = str(tmp_path / "art")
    assert inspect(src).weights == "unknown"          # the problem

    assert _script().main([src, "-o", out, "--weights", "ema"]) == 0

    art = inspect(out)
    assert art.fmt == "helix-eval"
    assert art.weights == "ema"                       # the fix
    assert art.op == OP
    assert art.arch["d"] == ARCH["d"]
    assert len(art.provenance["weights_digest"]) == 32
    assert art.provenance["source"] == os.path.abspath(src)


def test_the_weights_flag_is_not_optional(tmp_path):
    """The one thing only a human knows, asked once, at the moment it is known."""
    src = make_pimm_export(str(tmp_path / "exp"))
    with pytest.raises(SystemExit):
        _script().main([src, "-o", str(tmp_path / "art")])


def test_cell_t_is_required_but_pw_and_pt_are_defaulted(tmp_path):
    """The two halves of the operating point are NOT symmetric.

    `cell_t` has no default anywhere, because every possible one is wrong for
    someone and the failure is silent -- an artifact that cannot name it is the
    exact thing this format exists to prevent (94.06% of cells, mean |delta|
    19.5 ticks). So it is refused.

    `pw`/`pt` do have defaults, and the production configs rely on them: every
    real export records only `cell_t` in its CoeffTokenize cfg. Refusing there
    made the tool reject the thing it exists to serve. They are filled in and
    the fact is RECORDED, so the artifact still states a complete operating
    point and a default that changes later cannot move an already-scored number.
    """
    src = str(tmp_path / "exp")
    make_pimm_export(src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["transform"] = []                             # export without a tokenizer
    json.dump(cfg, open(os.path.join(src, "config.json"), "w"))

    with pytest.raises(SystemExit, match="records no tokenizer cell_t"):
        _script().main([src, "-o", str(tmp_path / "a"), "--weights", "ema"])

    out = str(tmp_path / "b")
    assert _script().main([src, "-o", out, "--weights", "ema",
                           "--cell-t", "grid_center"]) == 0
    art = inspect(out)
    assert (art.op.cell_t, art.op.pw, art.op.pt) == ("grid_center", 16, 8)
    assert art.provenance["operating_point_defaults"] == ["pt", "pw"], (
        "which fields came from a default must be on the record, or the "
        "artifact cannot be told apart from one whose run stated them")


def test_a_real_export_records_no_defaults(tmp_path):
    """The good case: nothing was guessed."""
    src = make_pimm_export(str(tmp_path / "exp"))     # carries pw/pt
    out = str(tmp_path / "art")
    _script().main([src, "-o", out, "--weights", "ema"])
    assert inspect(out).provenance["operating_point_defaults"] == []


def test_a_conflicting_cell_t_is_refused_not_resolved(tmp_path):
    src = make_pimm_export(str(tmp_path / "exp"))
    with pytest.raises(SystemExit, match="One of the two is wrong"):
        _script().main([src, "-o", str(tmp_path / "a"), "--weights", "raw",
                        "--cell-t", "centroid"])


def test_it_will_not_silently_replace_an_artifact(tmp_path):
    src = make_pimm_export(str(tmp_path / "exp"))
    out = str(tmp_path / "art")
    _script().main([src, "-o", out, "--weights", "ema"])
    with pytest.raises(SystemExit, match="already holds an artifact"):
        _script().main([src, "-o", out, "--weights", "raw"])
    assert inspect(out).weights == "ema", "the refused run must change nothing"
    _script().main([src, "-o", out, "--weights", "raw", "--force"])
    assert inspect(out).weights == "raw"


def test_the_recorded_digest_is_the_one_run_probe_recomputes(tmp_path):
    """ONE digest. There were briefly two, landing in the same results row.

    The artifact records it at promotion time over the SAVED tensors; run_probe
    recomputes it over the LOADED model. They must agree, which makes the
    comparison a real check on the load path rather than two unrelated hashes.
    """
    import importlib.util as _ilu
    import pathlib as _pl

    from helix.model.artifact import load, weights_digest

    src = make_pimm_export(str(tmp_path / "exp"))
    out = str(tmp_path / "art")
    _script().main([src, "-o", out, "--weights", "ema"])

    rp_path = _pl.Path(__file__).resolve().parents[1] / "scripts" / "run_probe.py"
    spec = _ilu.spec_from_file_location("_rp_digest_under_test", rp_path)
    rp = _ilu.module_from_spec(spec)
    spec.loader.exec_module(rp)

    from helix.probe.features import load_probe_model
    model, meta = load_probe_model(out, weights="ema", device="cpu")
    assert rp._weights_digest(model) == meta["provenance"]["weights_digest"]
    # and the export it was promoted from hashes identically: promotion moves
    # bytes, it does not change them
    assert weights_digest(load(src).state_dict) == meta["provenance"]["weights_digest"]


def test_the_artifact_and_the_probe_row_record_helix_the_SAME_way(tmp_path):
    """One schema for "which helix", checked across the two writers.

    The artifact stamped `coeff_io._code_version` -- a PRIVATE helper belonging
    to the corpus codec, returning {version, git, git_dirty} -- while the probe
    row stamped `_bootstrap.provenance()`, returning {root, commit, dirty,
    branch}. Both were called "helix"; no test compared them; run_probe read
    `.get("git")` off the artifact and would have silently produced an empty
    `ckpt_helix` the moment either side moved.

    `_code_version` deliberately stays as it is: corpus shards are verified
    field-by-field against that schema and must not change.
    """
    from helix.integrations._bootstrap import describe_checkout, running_roots

    src = make_pimm_export(str(tmp_path / "exp"))
    out = str(tmp_path / "art")
    _script().main([src, "-o", out, "--weights", "ema"])

    recorded = inspect(out).provenance["helix"]
    live = describe_checkout(running_roots()[0])
    assert set(recorded) == set(live) == {"root", "commit", "dirty", "branch"}
    assert recorded["commit"] == live["commit"]

    # and the key run_probe actually reads off it must be the one that is there
    import pathlib as _pl
    rp = (_pl.Path(__file__).resolve().parents[1] / "scripts" / "run_probe.py").read_text()
    assert 'get("commit", "")' in rp, "run_probe reads a key the artifact does not write"


def test_the_probe_loader_reads_a_promoted_artifact_and_attributes_it(tmp_path):
    """End to end: the number this produces is finally attributable."""
    from helix.probe.features import load_probe_model

    src = make_pimm_export(str(tmp_path / "exp"))
    out = str(tmp_path / "art")
    _script().main([src, "-o", out, "--weights", "ema"])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no "unattributed" warning
        model, meta = load_probe_model(out, weights="ema", device="cpu")
    assert meta["weights"] == "ema" and meta["weights_are_ema"] is True
    assert meta["tokenizer"]["cell_t"] == "grid_center"
    assert meta["provenance"]["weights_digest"]

    # And the same weights, byte for byte, as the export it was promoted from.
    exported, _ = load_probe_model(src, weights="raw", device="cpu")
    for k, v in exported.state_dict().items():
        torch.testing.assert_close(v, model.state_dict()[k], msg=k, equal_nan=True)
