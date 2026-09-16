"""The format matrix: every checkpoint shape, read by ONE loader.

There were eight shapes in circulation and eight independent ``torch.load``
sites, each of which had learned a subset. Nothing could catch the gaps because
nothing named the set — so a `pimm export` directory was taught to
``load_probe_model`` and not to ``patch_config_from_checkpoint``, and probing a
pimm-trained model died on ``IsADirectoryError``.

This file names the set. Every shape is either READ, with an operating point
that survives the round trip, or REFUSED by a message that says what to do
instead. A ninth shape added without a row here fails the last test.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm
from helix.model.artifact import (
    FORMATS, READABLE, Artifact, OperatingPoint, detect, inspect, load, save,
)

ARCH = dict(n_slot=8, n_band=4, n_plane=6, d=32, blocks=1, dec_blocks=1,
            heads=4, dec_mode="cross", n_bins=16)
#: The operating point every readable fixture below encodes. Identical across
#: formats on purpose: that is the property being tested.
OP = OperatingPoint(cell_t="grid_center", pw=16, pt=8, n_bands=4)
TOK = dict(cell_t="grid_center", pw=16, pt=8, n_bands=4)


def _model():
    m = build_fm(dict(ARCH))
    m.set_bins(torch.linspace(-4, 4, ARCH["n_bins"] + 1).repeat(ARCH["n_band"], 1))
    return m


# --------------------------------------------------------------- the fixtures

def make_pimm_export(d, *, weight=None):
    """What ``pimm export`` writes: weights + the resolved config beside them.

    The DDP ``module.`` prefix is part of the shape — pimm saves the wrapped
    model — and `weight` is None on every real export because
    ``_sanitize_config`` nulls it.
    """
    os.makedirs(d, exist_ok=True)
    torch.save({f"module.{k}": v for k, v in _model().state_dict().items()},
               os.path.join(d, "model.bin"))
    json.dump({"model": dict(ARCH, type="Coeff-FM", bins="<redacted>"),
               "weight": weight, "save_path": "<redacted>",
               "transform": [{"type": "CoeffTokenize", "cfg": dict(TOK)}]},
              open(os.path.join(d, "config.json"), "w"))
    return d


def make_helix_eval(d, *, weights="ema"):
    pytest.importorskip("safetensors")
    return save(d, state_dict=_model().state_dict(), arch=ARCH, op=OP,
                weights=weights, provenance={"helix_commit": "deadbeef"})


def make_converted(p):
    """The retired converter's output. Still on disk (m113's blob is), so it must
    still be DETECTED -- but it is no longer readable."""
    torch.save({"config": dict(ARCH), "tokenizer": dict(TOK),
                "state_dict": _model().state_dict(),
                "bins": {"edges": torch.linspace(-4, 4, 17).repeat(4, 1).tolist()},
                "provenance": {"weights": "ema", "source": "/somewhere/m113.pth"}},
               p)
    return p


def make_raw_state_dict(p):
    """pimm's ``model_ema.pth``: weights and a step, nothing else."""
    torch.save({"state_dict": _model().state_dict(), "step": 112677}, p)
    return p


def make_research(p):
    """The historical research shape. Nothing has been able to read it since the
    converter was retired; m113 was its only subject and now has an artifact."""
    torch.save({"model": _model().state_dict(), "ema": None, "step": 113000}, p)
    return p


def make_dcp_resume(d):
    """``<save_path>/model/last/``: RESUME state, not an eval artifact."""
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, ".metadata"), "wb").close()
    open(os.path.join(d, "__0_0.distcp"), "wb").close()
    return d


def make_bins_sidecar(p):
    torch.save({"edges": torch.linspace(-4, 4, 17).repeat(4, 1)}, p)
    return p


#: shape -> (builder, is-a-directory). The matrix itself.
MATRIX = {
    "pimm-export": (make_pimm_export, True),
    "helix-eval": (make_helix_eval, True),
    "converted": (make_converted, False),
    "raw-state-dict": (make_raw_state_dict, False),
    "research": (make_research, False),
    "dcp-resume": (make_dcp_resume, True),
    "unknown-dir": (lambda d: (os.makedirs(d, exist_ok=True), d)[1], True),
    "unknown-file": (make_bins_sidecar, False),
}


def _make(fmt, tmp_path):
    build, is_dir = MATRIX[fmt]
    return str(build(str(tmp_path / ("d_" + fmt if is_dir else fmt + ".pt"))))


@pytest.mark.parametrize("fmt", sorted(MATRIX))
def test_detect_names_every_shape(fmt, tmp_path):
    assert detect(_make(fmt, tmp_path)) == fmt


@pytest.mark.parametrize("fmt", ["pimm-export", "helix-eval"])
def test_the_operating_point_survives_every_readable_format(fmt, tmp_path):
    """THE property. A number scored through any of these is comparable.

    ``cell_t`` alone moves 94.06% of cells (mean |delta| 19.5 ticks), and
    scoring on the wrong one yields a plausible number rather than an error —
    so "the loader read it" is not enough; every loader must read the SAME one.
    """
    art = inspect(_make(fmt, tmp_path))
    assert art.op.comparable_to(OP), f"{fmt}: {art.op} != {OP}"
    assert art.arch["d"] == ARCH["d"] and art.arch["n_band"] == ARCH["n_band"]


@pytest.mark.parametrize("fmt", ["pimm-export", "helix-eval"])
def test_inspect_loads_no_weights(fmt, tmp_path):
    """Reading the tokenizer must not cost 236 MB of tensors."""
    assert inspect(_make(fmt, tmp_path)).state_dict is None
    assert load(_make(fmt, tmp_path)).state_dict is not None


@pytest.mark.parametrize("fmt,expect", [
    ("dcp-resume", "RESUME state"),
    ("raw-state-dict", "weights and nothing else"),
    ("converted", "was retired"),
    ("research", "was retired"),
    ("unknown-dir", "unrecognised"),
    ("unknown-file", "unrecognised"),
])
def test_what_cannot_be_scored_is_refused_by_name(fmt, expect, tmp_path):
    """Refusal must say WHICH shape it is and what to run instead.

    A DCP directory used to reach ``torch.load`` and die on IsADirectoryError,
    which tells the reader nothing about the fact that they pointed an eval tool
    at resume state.
    """
    with pytest.raises(ValueError, match=expect):
        inspect(_make(fmt, tmp_path))


def test_the_refusals_spell_the_one_remedy(tmp_path):
    """One remedy, spelled once. Four call sites each had their own copy."""
    for fmt in ("dcp-resume", "raw-state-dict"):
        with pytest.raises(ValueError) as e:
            inspect(_make(fmt, tmp_path))
        assert "pimm export --run-dir" in str(e.value)
        assert "export_artifact.py" in str(e.value), (
            "exporting is only half of it: an export cannot say which weight "
            "set it holds, so the remedy must name the promotion step too")


def test_a_retired_format_says_so_and_says_where_m113_went(tmp_path):
    """A retired reader must not degrade into "unrecognised".

    Both blobs still exist on disk -- m113's IS its own lineage, kept in the
    archive -- so someone will point a tool at one. Telling them the shape is
    unrecognised would be false; telling them the converter was retired and
    where m113 lives now is the whole value of detecting a format you cannot
    read.
    """
    for fmt in ("converted", "research"):
        with pytest.raises(ValueError) as e:
            inspect(_make(fmt, tmp_path))
        msg = str(e.value)
        assert "convert_fm_ckpt" in msg and "retired" in msg
        assert "fm_m113_artifact" in msg
        assert "git history" in msg, "say where it went, not just that it is gone"


def test_a_helix_eval_artifact_says_which_weights_it_holds(tmp_path):
    """The thing a `pimm export` structurally cannot do.

    ``_sanitize_config`` nulls `weight`, so every real export reads "unknown"
    and every probe row carries weights_are_ema=None. An artifact we write must
    answer it, and must refuse to be written without an answer.
    """
    assert inspect(make_helix_eval(str(tmp_path / "a"))).weights == "ema"
    assert inspect(make_pimm_export(str(tmp_path / "b"))).weights == "unknown"
    with pytest.raises(AssertionError, match="must say which weight set"):
        save(str(tmp_path / "c"), state_dict={}, arch=ARCH, op=OP,
             weights="unknown")


def test_inlined_bins_survive_the_round_trip_as_NUMBERS(tmp_path):
    """The edges are training-set statistics; nothing can re-derive them.

    `json.dump(..., default=str)` silently turned m113's edges into the string
    repr of a tensor. The artifact still wrote, still loaded, and still reported
    its operating point correctly -- it just could not build a model any more.
    A serialiser that cannot fail is one that loses data.
    """
    from helix.model.artifact import build

    m = build_fm(dict(ARCH))
    edges = torch.linspace(-4, 4, ARCH["n_bins"] + 1).repeat(ARCH["n_band"], 1)
    op = OperatingPoint(cell_t="grid_center", pw=16, pt=8, n_bands=4,
                        bins={"edges": edges})                # a TENSOR, as m113 had
    sd = {k: v for k, v in m.state_dict().items() if k != "bin_edges"}
    d = save(str(tmp_path / "a"), state_dict=sd, arch=ARCH, op=op, weights="raw")

    back = inspect(d).op.bins["edges"]
    assert isinstance(back, list) and isinstance(back[0][0], float), \
        f"edges came back as {type(back).__name__}: they were stringified"
    torch.testing.assert_close(torch.tensor(back), edges)
    # and the model builds from them, which is what they are FOR
    torch.testing.assert_close(build(load(d), device="cpu").bin_edges, edges)


def test_what_cannot_be_written_raises_instead_of_being_stringified(tmp_path):
    with pytest.raises(TypeError, match="cannot be written to an artifact"):
        save(str(tmp_path / "a"), state_dict={}, arch=dict(ARCH, film=object()),
             op=OP, weights="raw")


def test_film_comes_back_a_tuple_from_every_reader(tmp_path):
    """JSON has no tuple, and build_fm's behaviour depends on it."""
    arch = dict(ARCH, film=("band", "plane", "wire"))
    d = save(str(tmp_path / "a"), state_dict={}, arch=arch, op=OP, weights="raw")
    assert inspect(d).arch["film"] == ("band", "plane", "wire")


def test_a_helix_eval_artifact_round_trips_through_the_model(tmp_path):
    from helix.model.artifact import build
    pytest.importorskip("safetensors")
    m = _model()
    d = save(str(tmp_path / "a"), state_dict=m.state_dict(), arch=ARCH, op=OP,
             weights="ema")
    back = build(load(d), device="cpu")
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, back.state_dict()[k], msg=k, equal_nan=True)


def test_asking_for_ema_never_makes_a_weight_set_ema(tmp_path):
    """``pick`` reports what the artifact HOLDS; the request cannot change it.

    It used to be a selector, because the converted blobs carried `state_dict`
    and `state_dict_ema` side by side. Nothing writes two sets any more, so the
    only honest answer is what is there -- and for an export that is "unknown",
    never "raw", or an EMA arm and a raw arm compare in silence.
    """
    exported = load(make_pimm_export(str(tmp_path / "d")))
    assert exported.pick("ema")[1] == "unknown"
    assert exported.pick("raw")[1] == "unknown"

    promoted = load(make_helix_eval(str(tmp_path / "a"), weights="ema"))
    assert promoted.pick("raw")[1] == "ema", "asking for raw cannot unmake an EMA"

    raw = load(make_helix_eval(str(tmp_path / "b"), weights="raw"))
    assert raw.pick("ema")[1] == "raw", "asking for EMA cannot make one"


def test_every_declared_format_has_a_row(tmp_path):
    """Adding a shape to FORMATS without a fixture fails HERE, not in production."""
    assert set(MATRIX) == set(FORMATS)
    assert set(READABLE) <= set(FORMATS)


def test_nothing_readable_needs_a_retired_tool(tmp_path):
    """Every readable format is one the CURRENT toolchain can produce.

    That is what retiring the converter bought: two of the eight shapes were
    reachable only through a tool kept alive for one checkpoint.
    """
    assert set(READABLE) == {"helix-eval", "pimm-export"}
    for fmt in READABLE:
        assert inspect(_make(fmt, tmp_path)).op.recorded
