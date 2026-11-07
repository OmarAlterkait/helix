"""The simulation through the TPC chain (helix.tpc.chain SIM_BASE / sim_config): the layout
readout, the chain output pinned on synthetic planes, and the CoeffEvent / clean target."""
import numpy as np
import pytest

from helix.tpc.chain import TPCChain, sim_config
from helix.tpc import chain as tpc_chain
from helix.tpc.readout import load_readout


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
