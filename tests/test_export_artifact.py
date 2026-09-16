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
from tests.test_artifact_formats import ARCH, OP, make_pimm_export


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
    assert art.op.comparable_to(OP)
    assert art.arch["d"] == ARCH["d"]
    assert len(art.provenance["weights_digest"]) == 32
    assert art.provenance["source"] == os.path.abspath(src)


def test_the_weights_flag_is_not_optional(tmp_path):
    """The one thing only a human knows, asked once, at the moment it is known."""
    src = make_pimm_export(str(tmp_path / "exp"))
    with pytest.raises(SystemExit):
        _script().main([src, "-o", str(tmp_path / "art")])


def test_an_artifact_without_cell_t_is_refused(tmp_path):
    """Refusing to write is the whole point.

    An artifact whose cell_t is unknown is the exact failure this format exists
    to prevent: the probe falls back to a default and 94.06% of cells carry a
    time coordinate the model never saw.
    """
    src = str(tmp_path / "exp")
    make_pimm_export(src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["transform"] = []                             # export without a tokenizer
    json.dump(cfg, open(os.path.join(src, "config.json"), "w"))

    with pytest.raises(SystemExit, match="records no tokenizer cell_t"):
        _script().main([src, "-o", str(tmp_path / "a"), "--weights", "ema"])

    # ...and --cell-t supplies it, because pw/pt being absent too is a DIFFERENT
    # failure with a different remedy.
    with pytest.raises(SystemExit, match="operating point is incomplete"):
        _script().main([src, "-o", str(tmp_path / "b"), "--weights", "ema",
                        "--cell-t", "grid_center"])


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
