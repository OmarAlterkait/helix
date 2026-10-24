"""The configurable TPC chain (helix.tpc.chain) and its CoeffEvents (helix.tpc.chain_coeff)."""
import numpy as np
import pytest

from helix.tpc import chain as ch
from helix.tpc import chain_coeff
from helix.tpc.readout import load_readout


def _event(nt=512, seed=0):
    ro = load_readout("sbnd")
    rng = np.random.default_rng(seed)
    n = len(ro.tpc)
    ped = np.where(ro.plane == 2, 400.0, 2048.0).astype(np.float32)
    adc = np.round(ped[:, None] + rng.normal(0, 2, (n, nt))).astype(np.int16)
    rows = ro.rows(0)
    adc[rows[2][100:104], 200:206] += 60                 # a collection deposit
    adc[rows[0][300], :] = ped[rows[0][300]]              # a dead channel
    present = ro.instrumented.copy()
    return adc, ped, present


def test_config_defaults_overrides_and_digest():
    base = ch.make_config()
    assert base["threshold"]["kappa"] == 1.0 and base["threshold"]["floor"] == 2.0 and base["dwt"]["level"] == 4
    with pytest.raises(ValueError):                                          # one basis for all planes
        ch.make_config({"dwt": {"level": {"U": 4, "V": 3, "Y": 3}}})
    c = ch.make_config({"threshold": {"kappa": {"U": 0.8, "V": 1.0, "Y": 0.9}}})
    assert ch.per_plane(c["threshold"]["kappa"], "V") == 1.0
    assert ch.config_digest(c) != ch.config_digest(base)
    with pytest.raises(ValueError):
        ch.make_config({"threshold": {"rule": "nope"}})
    with pytest.raises(ValueError):
        ch.make_config({"threshold": {"kapa": 1.0}})


@pytest.mark.parametrize("overrides", [None, {"threshold": {"floor": None, "kappa": 1.1}},
                                       {"coherent": {"groups": None}, "threshold": {"floor": None}},
                                       {"coherent": {"cross_plane": False}, "threshold": {"drop_finest": 0}},
                                       {"coherent": {"groups": "asic", "tau": 0.05, "sigc_scope": "global"}}])
def test_chain_runs_and_excludes_dead_rows(overrides):
    adc, ped, present = _event()
    chain = ch.TPCChain(overrides)
    res = chain.process_event(adc, ped, present)
    assert len(res) == 6
    y0 = [r for r in res if r["tpc"] == 0 and r["plane"] == "Y"][0]
    assert y0["n_kept"] > 0
    u0 = [r for r in res if r["tpc"] == 0 and r["plane"] == "U"][0]
    assert not u0["valid"][300]
    assert all(not b[300].any() for b in u0["bands"])
    nd = chain.cfg["threshold"]["drop_finest"]
    for r in res:
        assert r["level"] == chain.cfg["dwt"]["level"]
        assert all(not b.any() for b in r["bands"][len(r["bands"]) - nd:])


def test_baseline_step_row_is_excluded():
    adc, ped, present = _event(seed=2)
    rows = load_readout("sbnd").rows(0)
    adc[rows[1][40], 300:] += 40                          # a 20-rms baseline step on one V channel
    res = ch.TPCChain().process_event(adc, ped, present)
    v0 = [r for r in res if r["tpc"] == 0 and r["plane"] == "V"][0]
    assert not v0["valid"][40] and v0["valid"][41]


def test_hysteresis_keeps_a_faint_track_end_universal_drops():
    """A track fading to ~2.5 sigma per coefficient: hysteresis follows it from its bright
    part; the universal threshold (same kappa) keeps only the bright part."""
    adc, ped, present = _event(seed=3)
    rows = load_readout("sbnd").rows(0)[2]
    amp = np.linspace(40, 4, 120)
    for k in range(120):                                  # a diagonal track on Y, fading
        adc[rows[500 + k], 100 + 2 * k:104 + 2 * k] += np.int16(amp[k])
    kept = {}
    for rule, floor in (("universal", None), ("hysteresis", 2.0)):
        res = ch.TPCChain({"threshold": {"floor": floor, "kappa": 1.1}}).process_event(adc, ped, present)
        y0 = [r for r in res if r["tpc"] == 0 and r["plane"] == "Y"][0]
        kept[rule] = sum(int(np.count_nonzero(b[580:620])) for b in y0["bands"])
    assert kept["hysteresis"] > 1.5 * kept["universal"]


def test_coeff_shard_roundtrip(tmp_path):
    """Chain results -> CoeffEvent -> coeff_io shard -> the same coefficients."""
    from helix.core import coeff_io
    chain = ch.TPCChain()
    ces = [chain_coeff.coeff_event(chain, chain.process_event(*_event(seed=1 + e)), run="1", source_file="x.h5", event=7 + e)
           for e in range(3)]
    p = tmp_path / "x.h5"
    coeff_io.write_coeff_shard(p, ces)
    coeff_io.audit_shard(p)
    for i, ce in enumerate(ces):
        back = coeff_io.read_coeff_event(p, i)
        for k in ("band", "plane_gid", "wire", "tau", "value"):
            np.testing.assert_array_equal(getattr(back, k), getattr(ce, k))
        assert back.event == 7 + i


def test_coeff_event_physical_wires_and_validity(tmp_path):
    """SBND chain -> CoeffEvent: physical wire numbers (uninstrumented wires are
    gaps), per-wire validity through the shard, and the tokenizer gives excluded wires
    no target."""
    from helix.core import coeff_io
    from helix.model.tokenize import PatchConfig, assemble
    ro = load_readout("sbnd")
    adc, ped, present = _event(seed=4)
    chain = ch.TPCChain()
    res = chain.process_event(adc, ped, present)
    ce = chain_coeff.coeff_event(chain, res, run="1", source_file="s.h5", event=3)
    y0 = int(np.where(ce.gids == 2)[0][0])
    yw = ro.wire[(ro.tpc == 0) & (ro.plane == 2)]
    assert ce.n_wires[y0] == yw.max() + 1
    gap = np.setdiff1d(np.arange(yw.max() + 1), ro.wire[ro.rows(0)[2]])
    assert len(gap) > 0 and not ce.wire_valid[y0, gap].any()
    u0 = int(np.where(ce.gids == 0)[0][0])
    dead = ro.wire[ro.rows(0)[0][300]]
    assert not ce.wire_valid[u0, dead] and ce.wire_valid[u0, dead + 1]
    for r in res:                                        # same coefficients, physical wire coordinates
        g = r["tpc"] * 3 + "UVY".index(r["plane"])
        assert int((ce.plane_gid == g).sum()) == int(r["n_kept"])
    coeff_io.write_coeff_shard(tmp_path / "c.h5", [ce])
    back = coeff_io.read_coeff_event(tmp_path / "c.h5", 0)
    assert np.array_equal(back.wire_valid, ce.wire_valid) and np.array_equal(back.wire, ce.wire)
    kw = dict(gids=ce.gids, n_wires=ce.n_wires, band_lengths=ce.basis.band_lengths,
              norm_sigma=np.maximum(ce.sigma_threshold, 1e-3), cfg=PatchConfig(cell_t="grid_center"))
    a = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, **kw)
    b = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, wire_valid=ce.wire_valid, **kw)
    assert a["n_cells"] == b["n_cells"] and b["valid"].sum() < a["valid"].sum()
    assert not (b["valid"] & ~a["valid"]).any()
