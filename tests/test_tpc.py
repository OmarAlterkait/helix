"""helix.tpc: the TPC chain and its CoeffEvents, the simulation through it, thresholding, the
batched torch ops on SBND-like planes, and the JAXTPC campaign reader with the simulation source."""
import pytest

torch = pytest.importorskip("torch")

import os

import numpy as np
import pywt
import h5py

from helix.tpc import chain as tpc_chain
from helix.tpc.readout import load_readout
from helix.tpc.chain import TPCChain, sim_config
from helix.tpc import threshold_rows as tr  # noqa: E402
from _fixtures import CAMPAIGN, N_TICKS, N_WIRES, PEDESTAL, PLANES, write_geometry, write_task
from helix.tpc import io
from _fixtures import sim_file                                     # noqa: E402


# ---- the TPC chain (helix.tpc.chain) -----------------------------------------------------

# The configurable TPC chain (helix.tpc.chain) and its CoeffEvents.

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
    base = tpc_chain.make_config()
    assert base["threshold"]["kappa"] == 1.0 and base["threshold"]["floor"] == 2.0 and base["dwt"]["level"] == 4
    with pytest.raises(ValueError):                                          # one basis for all planes
        tpc_chain.make_config({"dwt": {"level": {"U": 4, "V": 3, "Y": 3}}})
    c = tpc_chain.make_config({"threshold": {"kappa": {"U": 0.8, "V": 1.0, "Y": 0.9}}})
    assert tpc_chain.per_plane(c["threshold"]["kappa"], "V") == 1.0
    assert tpc_chain.config_digest(c) != tpc_chain.config_digest(base)
    with pytest.raises(ValueError):
        tpc_chain.make_config({"threshold": {"rule": "nope"}})
    with pytest.raises(ValueError):
        tpc_chain.make_config({"threshold": {"kapa": 1.0}})


@pytest.mark.parametrize("overrides", [None, {"threshold": {"floor": None, "kappa": 1.1}},
                                       {"coherent": {"groups": None}, "threshold": {"floor": None}},
                                       {"coherent": {"cross_plane": False}, "threshold": {"drop_finest": 0}},
                                       {"coherent": {"groups": "asic", "tau": 0.05, "sigc_scope": "global"}}])
def test_chain_runs_and_excludes_dead_rows(overrides):
    adc, ped, present = _event()
    chain = tpc_chain.TPCChain(overrides)
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
    res = tpc_chain.TPCChain().process_event(adc, ped, present)
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
        res = tpc_chain.TPCChain({"threshold": {"floor": floor, "kappa": 1.1}}).process_event(adc, ped, present)
        y0 = [r for r in res if r["tpc"] == 0 and r["plane"] == "Y"][0]
        kept[rule] = sum(int(np.count_nonzero(b[580:620])) for b in y0["bands"])
    assert kept["hysteresis"] > 1.5 * kept["universal"]


def test_coeff_shard_roundtrip(tmp_path):
    """Chain results -> CoeffEvent -> coeff_io shard -> the same coefficients."""
    from helix.core import coeff_io
    chain = tpc_chain.TPCChain()
    ces = [tpc_chain.coeff_event(chain, chain.process_event(*_event(seed=1 + e)), run="1", source_file="x.h5", event=7 + e)
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
    chain = tpc_chain.TPCChain()
    res = chain.process_event(adc, ped, present)
    ce = tpc_chain.coeff_event(chain, res, run="1", source_file="s.h5", event=3)
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
              norm_sigma=np.maximum(ce.sigma_threshold, 1e-3), cfg=PatchConfig())
    a = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, **kw)
    b = assemble(ce.band, ce.plane_gid, ce.wire, ce.tau, ce.value, wire_valid=ce.wire_valid, **kw)
    assert a["n_cells"] == b["n_cells"] and b["valid"].sum() < a["valid"].sum()
    assert not (b["valid"] & ~a["valid"]).any()


# ---- the simulation through the chain ----------------------------------------------------

# The simulation through the TPC chain (helix.tpc.chain SIM_BASE / sim_config): the layout
# readout, the chain output pinned on synthetic planes, and the CoeffEvent / clean target.

def _planes(seed=0, nw=(200, 200, 150), nt=1000):
    """Noise-free tracks and noisy images: white noise 2.6 ADC + a coherent waveform per
    64-wire block (the simulation's noise model, simplified)."""
    rng = np.random.default_rng(seed)
    clean, noisy = [], []
    for n in nw:
        c = np.zeros((n, nt), np.float32)
        for _ in range(4):
            w0, t0, sl = rng.integers(0, n - 40), rng.integers(100, nt - 200), rng.uniform(0.5, 3)
            for k in range(40):
                t = int(t0 + sl * k); c[w0 + k, t:t + 12] += 40 * np.exp(-np.arange(12) / 4.0)
        x = c + rng.normal(0, 2.6, c.shape)
        for b in range(0, n, 64):
            x[b:b + 64] += np.cumsum(rng.normal(0, 0.3, nt))[None, :]
        clean.append(c); noisy.append(x.astype(np.float32))
    return clean, noisy


def test_layout_readout():
    ro = load_readout({"kind": "layout", "num_wires": [[10, 10, 7], [10, 10, 7]], "group_size": 4})
    assert ro.n_tpc == 2 and ro.planes == ["U", "V", "Y"] and len(ro.tpc) == 54
    rows = ro.rows(1)
    assert [len(r) for r in rows] == [10, 10, 7] and (ro.wire[rows[2]] == np.arange(7)).all()
    lab = ro.labels("block", rows[0]); assert (np.diff(lab) >= 0).all() and len(np.unique(lab)) == 3


#: The chain's output on ``_planes()`` with D1 kept (numpy), pinned when the chain was
#: validated against the earlier simulation path (the retired per-plane pipeline with
#: the gate: equal to rtol 1e-5 / atol 1e-4 on every band). That path is gone;
#: these numbers keep the parity it established.
_PINNED_KEPT = [[389, 394, 397, 420, 316], [407, 416, 420, 403, 315], [408, 390, 381, 442, 332]]
_PINNED_ABS_SUM = [30274.419497966766, 30496.596704006195, 30297.72803211212]
_PINNED_POSITIONS = "1fd2b2f6f05f3f2eb512a353311c86311c0801a56046c167eb3e511a17b0eed6"


def test_chain_output_is_pinned_on_the_synthetic_planes():
    """Same gate, threshold and basis as the earlier simulation path: kept count per
    band, a hash of the kept positions, and the kept magnitude are unchanged (numpy)."""
    import hashlib
    clean, noisy = _planes()
    ch = TPCChain(sim_config([[len(x) for x in noisy]], {"threshold": {"drop_finest": 0}}))
    res = ch.process_images(noisy, tpc=0)
    h = hashlib.sha256()
    kept, abs_sum = [], []
    for r in res:
        bands = [np.asarray(b) for b in r["bands"]]
        kept.append([int(np.count_nonzero(b)) for b in bands])
        abs_sum.append(float(sum(np.abs(b.astype(np.float64)).sum() for b in bands)))
        for b in bands:
            idx = np.flatnonzero(b).astype(np.int64)
            h.update(np.int64(idx.size).tobytes()); h.update(idx.tobytes())
    assert kept == _PINNED_KEPT
    assert [r["n_kept"] for r in res] == [sum(k) for k in _PINNED_KEPT]
    assert h.hexdigest() == _PINNED_POSITIONS
    np.testing.assert_allclose(abs_sum, _PINNED_ABS_SUM, rtol=1e-5)


def test_sim_coeff_event_and_clean_target():
    clean, noisy = _planes(seed=1)
    ch = TPCChain(sim_config([[len(x) for x in noisy]]))
    res = ch.process_images(noisy, tpc=0)
    ce = tpc_chain.coeff_event(ch, res, run="r", source_file="f.h5", event=3)
    assert list(ce.gids) == [0, 1, 2] and list(ce.n_wires) == [200, 200, 150] and ce.wire_valid is None
    assert len(ce.value) > 0 and (ce.band < 4).all()                      # D1 dropped (drop_finest 1)
    cl = tpc_chain.clean_coeff_event(ch, ce, {g: clean[g] for g in range(3)})
    assert np.array_equal(cl.wire, ce.wire) and cl.basis == ce.basis
    big = np.abs(ce.value) > 20                                           # strong signal: noisy ~ clean
    assert np.median(np.abs(ce.value[big] - cl.value[big]) / np.abs(ce.value[big])) < 0.2


def test_stored_values_on_their_step():
    """Input and target values lie on a power-of-two grid <= sigma / q of their (plane, band),
    within half a step of the exact values; q = 0 stores them exact."""
    clean, noisy = _planes(seed=1)
    exact = TPCChain(sim_config([[len(x) for x in noisy]], {"store": {"step_input": 0, "step_target": 0}}))
    ch = TPCChain(sim_config([[len(x) for x in noisy]]))
    res = ch.process_images(noisy, tpc=0)
    ce0, ce = (tpc_chain.coeff_event(c, res) for c in (exact, ch))
    cl0, cl = (tpc_chain.clean_coeff_event(c, e, {g: clean[g] for g in range(3)}) for c, e in ((exact, ce0), (ch, ce)))
    assert ce.basis.threshold["value_step"] == {"step_input": 16, "step_target": 32}
    assert ce.basis.digest() != ce0.basis.digest()
    for q, x0, x in ((16, ce0, ce), (32, cl0, cl)):
        step = tpc_chain.value_step(ce.sigma_threshold, q)[np.searchsorted(ce.gids, ce.plane_gid), ce.band].astype(np.float64)
        sig = ce.sigma_threshold[np.searchsorted(ce.gids, ce.plane_gid), ce.band]
        assert (step > 0).all() and (step <= sig / q).all() and (step > sig / (2 * q)).all()
        np.testing.assert_array_equal(np.mod(x.value / step, 1), 0)
        assert (np.abs(x.value.astype(np.float64) - x0.value) <= step / 2).all()
        assert x.value.dtype == np.float32


# ---- thresholding (helix.tpc.threshold_rows) ---------------------------------------------

# Live-row threshold: parity with the universal threshold and robustness to missing channels.

def threshold_bands_rows(coeffs, **kw):
    out, k, n, s = tr.threshold_bands_rows(coeffs, **kw)
    return [o.numpy() for o in out], k, n, s


def _universal(bands):
    """HELIX's universal rule in numpy: per-band MAD sigma over all rows, every band
    (approximation too) hard-thresholded at sigma sqrt(2 ln N_band)."""
    out = []
    for c in bands:
        s = np.float32(np.median(np.abs(c)) / 0.6745)
        out.append(np.where(np.abs(c) >= s * np.float32(np.sqrt(2 * np.log(c.shape[-1]))), c, 0).astype(np.float32))
    return out, int(sum(np.count_nonzero(o) for o in out))


def _bands(seed=0, rows=400):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 2, (rows, 1024)).astype(np.float32)
    x[50:60, 400:420] += 30
    return [np.asarray(c, np.float32) for c in pywt.wavedec(x, "coif3", level=4, mode="periodization", axis=-1)]


def test_parity_with_threshold_bands_all_rows_valid():
    b = _bands()
    ref, kref = _universal(b)
    got, kgot, _, _ = threshold_bands_rows(b, kappa=1.0, floor=None)          # seeds only = universal
    assert abs(kgot - kref) <= 2                              # float32 sigma: only values on the threshold flip
    for r, g in zip(ref, got):
        both = (r != 0) & (g != 0)
        assert both.sum() >= (r != 0).sum() - 2
        np.testing.assert_array_equal(r[both], g[both])


def test_missing_rows_do_not_collapse_sigma():
    b = _bands(seed=1)
    valid = np.ones(400, bool); valid[:250] = False            # most of the plane missing
    zeroed = [np.where(valid[:, None], c, 0.0).astype(np.float32) for c in b]
    _, k_all_rows = _universal(zeroed)
    _, k_live, _, _ = threshold_bands_rows(zeroed, kappa=1.0, valid=valid)
    _, k_ref, _, _ = threshold_bands_rows([c[valid] for c in b], kappa=1.0)
    assert k_all_rows > 20 * k_live                           # the bug: sigma from mostly-zero rows
    assert k_live == k_ref                                    # the fix: same as the live rows alone


def test_group_and_channel_scope():
    b = _bands(seed=2)
    groups = np.arange(400) // 100
    for scope, g in (("group", groups), ("channel", None)):
        out, k, _, s = threshold_bands_rows(b, kappa=1.0, sigma_scope=scope, groups=g)
        assert 0 < k < sum(c.size for c in b) and np.all(np.isfinite(s[1:]))


def test_hysteresis_is_a_superset_grown_through_the_floor():
    b = _bands(seed=3)
    plain, kp, _, _ = threshold_bands_rows(b, kappa=1.1, floor=None)
    hyst, kh, _, _ = threshold_bands_rows(b, kappa=1.1, floor=2.0)
    assert kh > kp
    for p, h in zip(plain, hyst):
        assert np.array_equal(h[p != 0], p[p != 0])        # every plain survivor is kept
    # a floor above the seed threshold adds nothing
    same, ks, _, _ = threshold_bands_rows(b, kappa=1.1, floor=50.0)
    assert ks == kp


def test_sim_chain_hysteresis_default():
    """The simulation chain thresholds with hysteresis by default: a superset of the
    seeds-only (old universal) survivors, and recorded in the chain digest."""
    from helix.tpc.chain import TPCChain, sim_config
    rng = np.random.default_rng(4)
    img = rng.normal(0, 2, (128, 2701)).astype(np.float32)
    img[40:60, 1000:1012] += 25
    nw = [[128, 128, 128]]
    ch = TPCChain(sim_config(nw, {"coherent": {"groups": None}}))
    cu = TPCChain(sim_config(nw, {"coherent": {"groups": None}, "threshold": {"floor": None}}))
    assert ch.cfg["threshold"]["floor"] == 2.0
    ph = ch.process_images([img] * 3, tpc=0)[0]
    pu = cu.process_images([img] * 3, tpc=0)[0]
    assert ph["n_kept"] > pu["n_kept"]
    for a, b in zip(pu["bands"], ph["bands"]):
        a, b = a.numpy(), b.numpy()
        assert np.array_equal(b[a != 0], a[a != 0])
    assert ch.digest != cu.digest


# ---- all planes at once vs one at a time -------------------------------------------------

# Fast paths agree with their general forms, on CPU and (when available) CUDA (run by hand on
# turing: CI has no GPU): all planes' hysteresis grown at once vs one plane at a time.

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _t(x, dev):
    return torch.as_tensor(np.asarray(x), device=dev)


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _sbnd_planes(seed=0, nt=512, rows=(96, 96, 64)):
    """Three planes with board common modes shared across planes, dead rows, pulses."""
    rng = np.random.default_rng(seed)
    labels = [np.arange(r) // 32 for r in rows]                        # 3 boards per plane span all planes
    cm = rng.normal(0, 3, (4, nt)).astype(np.float32) * rng.uniform(0.5, 3, (4, 1)).astype(np.float32)
    imgs = []
    for p, r in enumerate(rows):
        x = rng.normal(0, 2, (r, nt)).astype(np.float32) + cm[labels[p]]
        x[10:14, 200:210] += 40.0                                       # a track segment
        x[50, 300:306] += 12.0                                          # a small deposit
        dead = [3, 70] if r > 70 else [3]
        x[dead] = 0.0                                                   # dead rows
        labels[p] = labels[p].copy(); labels[p][dead] = -1
        imgs.append(x)
    return imgs, labels


def _sbnd_bands(img, wav="coif3", level=4):
    return [np.asarray(c, np.float32) for c in pywt.wavedec(img, wav, level=level, mode="periodization", axis=-1)]


@pytest.mark.parametrize("dev", DEVICES)
def test_threshold_planes_rows_matches_per_plane(dev):
    imgs, labels = _sbnd_planes(seed=6)
    jobs = [([_t(c, dev) for c in _sbnd_bands(im)], dict(kappa=1.1, valid=_t(lab >= 0, dev), floor=2.0))
            for im, lab in zip(imgs, labels)]
    together = tr.threshold_planes_rows(jobs)
    for (b, kw), (o, k, n, s) in zip(jobs, together):
        o1, k1, n1, s1 = tr.threshold_bands_rows(b, **kw)
        assert (k, n) == (k1, n1) and np.array_equal(s, s1)
        for x, y in zip(o, o1):
            assert torch.equal(x, y)


# ---- the campaign reader (helix.tpc.io) and SimSource ------------------------------------

# helix.tpc.io and the simulation source (helix.build.SimSource) on a JAXTPC wire campaign.
#
# Synthetic tasks in the campaign's layout and format (tests/_fixtures.py, format 2: entry_NNNNNN,
# /dprod/events, campaign event ids), written uncompressed; production files additionally need
# hdf5plugin, which io imports best-effort. Then the real campaign task, site-gated.

def _source():
    from helix.build import SimSource
    return SimSource


def test_layout_from_config(tmp_path):
    t = write_task(str(tmp_path))
    lay = io.sensor_layout(t["sensor"])
    assert lay.num_time_steps == N_TICKS and lay.campaign == CAMPAIGN
    assert lay.plane_labels == tuple(f"volume_{v}_{p}" for v in range(2) for p in PLANES)
    assert lay.pedestals == PEDESTAL
    assert [lay.n_wires(f"volume_1_{p}") for p in PLANES] == [N_WIRES[p] for p in PLANES]
    assert [lay.plane_gid(lab) for lab in lay.plane_labels] == list(range(6))


def test_event_roundtrip_with_an_empty_volume(tmp_path):
    """Every plane decodes to its pedestal-subtracted truth at the config's shape; a volume with
    nothing deposited (empty plane groups, no datasets, no attrs) is zero hits, and its neighbour
    keeps its charge."""
    t = write_task(str(tmp_path), n_events=3, empty={(1, 0)})
    lay = io.sensor_layout(t["sensor"])
    for pos in range(3):
        planes = io.read_event(t["sensor"], pos, lay)
        assert set(planes) == set(lay.plane_labels)
        for lab, img in planes.items():
            np.testing.assert_array_equal(img, t["truth"][(pos, lab)])
    with h5py.File(t["sensor"], "r") as f:
        coo = io.read_event_coo(f[io.entry_key(1)], lay)
    assert all(coo[f"volume_0_{p}"][0].size == 0 for p in PLANES)
    assert all(coo[f"volume_1_{p}"][0].size > 0 and coo[f"volume_1_{p}"][2].min() >= 3 for p in PLANES)


def test_entries_are_checked_against_the_campaign_identity(tmp_path):
    t = write_task(str(tmp_path), first_job=85, n_events=3)
    with h5py.File(t["sensor"], "r") as f:
        ids = io.event_ids(f)
        assert ids.tolist() == [17000, 17001, 17002]
        assert io.entry(f, 2, 17002).name == "/entry_000002"
        with pytest.raises(ValueError, match="dprod"):
            io.entry(f, 2, 17001)


def test_format_1_and_pixel_files_are_refused(tmp_path):
    p = tmp_path / "old.h5"
    with h5py.File(p, "w") as f:
        f.create_group("config").attrs["num_time_steps"] = 4321
        f.create_group("event_000")
    with pytest.raises(ValueError, match="format-2"):
        io.sensor_layout(p)
    t = write_task(str(tmp_path))
    with h5py.File(t["sensor"], "a") as f:
        f["config"].attrs["readout_type"] = "pixel"
    with pytest.raises(ValueError, match="pixel"):
        io.sensor_layout(t["sensor"])


def test_task_paths(tmp_path):
    t = write_task(str(tmp_path), group="j001000", first_job=1085)
    assert io.task_files(t["sensor"]) == {k: t[k] for k in ("sensor", "step", "hits")}
    assert io.sensor_path(str(tmp_path), "j001000", os.path.basename(t["sensor"])) == t["sensor"]


# ------------------------------------------------------------------------------- the build source
def test_sim_source_events_carry_the_campaign_identity(tmp_path):
    """The source's events: ident (group, file, campaign event id), the noise seed from it, every
    plane present (the empty volume's as zeros), the clean planes the file's own."""
    pytest.importorskip("pimm_data")
    Sim = _source()
    t = write_task(str(tmp_path), first_job=85, n_events=3, empty={(1, 1)})
    src = Sim([t["sensor"]], "cpu", geom=write_geometry(tmp_path / "geom.json"))
    src.group_size = 8
    assert len(src) == 3 and src.run == "j000000" and src.campaign == CAMPAIGN
    evs = list(src.events(range(3), 1))
    name = os.path.basename(t["sensor"])
    assert [e["ident"]["event"] for e in evs] == [17000, 17001, 17002]
    assert {e["ident"]["run"] for e in evs} == {"j000000"} and {e["ident"]["source_file"] for e in evs} == {name}
    assert [e["ident"]["noise_seed"] for e in evs] == [src.seed(t["sensor"], ev) for ev in (17000, 17001, 17002)]
    assert len({e["ident"]["noise_seed"] for e in evs}) == 3
    lay = io.sensor_layout(t["sensor"])
    for pos, e in enumerate(evs):
        assert set(e["planes"]) == set(e["clean"]) == set(range(6))
        for lab in lay.plane_labels:
            g = lay.plane_gid(lab)
            np.testing.assert_array_equal(e["clean"][g].numpy(), t["truth"][(pos, lab)])
            assert e["planes"][g].shape == (lay.n_wires(lab), N_TICKS)
            assert float(e["planes"][g].std()) > 0                 # noise is there, empty plane or not
    assert all(not e.any() for g, e in evs[1]["clean"].items() if g >= 3)
    attrs = src.shard_attrs()["provenance"]
    assert attrs["sim"] == dict(campaign=CAMPAIGN, jaxtpc=["abc"])


def test_sim_source_refuses_mixed_groups_and_a_wrong_geometry(tmp_path):
    Sim = _source()
    a = write_task(str(tmp_path), group="j000000", first_job=0)
    b = write_task(str(tmp_path), group="j001000", first_job=1000)
    geom = write_geometry(tmp_path / "geom.json")
    with pytest.raises(SystemExit, match="groups"):
        Sim([a["sensor"], b["sensor"]], None, geom=geom)
    with pytest.raises(SystemExit, match="plane"):
        Sim([a["sensor"]], None)                   # the packaged geometry is the full-size detector


def test_plan_numbers_shards_by_campaign_event(tmp_path):
    """One task per sensor file; shard k holds campaign events k*eps .. (k+1)*eps-1, so a task of
    jobs 85-89 (events 17000-17999) writes shards 85-89 at 200 per shard."""
    from helix.build import plan
    Sim = _source()
    root = tmp_path / "sim"
    t0 = write_task(str(root), first_job=0, n_events=1000, n_tracks=1, n_volumes=1)
    t1 = write_task(str(root), first_job=85, n_events=1000, n_tracks=1, n_volumes=1)
    lines = plan(Sim, "OUT", [str(root / "j000000")], events_per_task=500, events_per_shard=200)
    assert lines == [f"OUT/j000000/sim_wire 0 1000 200 0 {t0['sensor']}",
                     f"OUT/j000000/sim_wire 0 1000 200 85 {t1['sensor']}"]
    with pytest.raises(ValueError, match="multiple of 300"):
        Sim.file_plan(t1["sensor"], 300)


# ------------------------------------------------------------------------------ the real campaign

REAL = sim_file("sensor")


@pytest.mark.skipif(not os.path.exists(REAL), reason="campaign task not reachable")
def test_real_task_reads():
    """The campaign's task j000085-000089: 1000 events, ids 17000-17999, the detector's layout;
    entry 202 has nothing in volume 1."""
    lay = io.sensor_layout(REAL)
    assert lay.num_time_steps == 4321 and lay.num_wires.tolist() == [[1969, 1969, 1443]] * 2
    assert lay.pedestals == {"U": 1843, "V": 1843, "Y": 410}
    assert lay.campaign == "doraemon_2026_1M_lartpc"
    with h5py.File(REAL, "r") as f:
        ids = io.event_ids(f)
        assert ids.tolist() == list(range(17000, 18000))
        coo = io.read_event_coo(io.entry(f, 202, 17202), lay)
    assert all(coo[f"volume_1_{p}"][0].size == 0 for p in PLANES)
    for p in PLANES:
        w, t, v = coo[f"volume_0_{p}"]
        assert w.size > 1000 and 0 <= w.min() and w.max() < lay.n_wires(f"volume_0_{p}")
        assert 0 <= t.min() and t.max() < 4321 and np.abs(v).min() >= 1.0     # 2-ADC zero suppression
