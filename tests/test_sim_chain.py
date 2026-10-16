"""The simulation through the TPC chain (helix.tpc.chain SIM_BASE / sim_config): the layout
readout, parity with the earlier simulation path, and the CoeffEvent / clean target."""
import numpy as np
import pytest

from helix.tpc.chain import TPCChain, sim_config
from helix.tpc import chain_coeff
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


def test_chain_matches_the_earlier_simulation_path():
    """Same gate, threshold and basis: the chain with D1 kept equals process_plane(gate) (numpy)."""
    from helix.tpc.config import DetectorConfig
    from helix.tpc.pipeline import process_plane
    clean, noisy = _planes()
    ch = TPCChain(sim_config([[len(x) for x in noisy]], {"threshold": {"drop_finest": 0}}))
    res = ch.process_images(noisy, tpc=0)
    cfg = DetectorConfig(num_time_steps=noisy[0].shape[1])
    for x, r in zip(noisy, res):
        ref = process_plane(x, cfg, removal="gate", with_images=False).sparse.coeffs
        assert len(ref) == len(r["bands"])
        for a, b in zip(ref, r["bands"]):
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-5, atol=1e-4)


def test_sim_coeff_event_and_clean_target():
    clean, noisy = _planes(seed=1)
    ch = TPCChain(sim_config([[len(x) for x in noisy]]))
    res = ch.process_images(noisy, tpc=0)
    ce = chain_coeff.coeff_event(ch, res, run="r", source_file="f.h5", event=3)
    assert list(ce.gids) == [0, 1, 2] and list(ce.n_wires) == [200, 200, 150] and ce.wire_valid is None
    assert len(ce.value) > 0 and (ce.band < 4).all()                      # D1 dropped (drop_finest 1)
    cl = chain_coeff.clean_coeff_event(ch, ce, {g: clean[g] for g in range(3)})
    assert np.array_equal(cl.wire, ce.wire) and cl.basis == ce.basis
    big = np.abs(ce.value) > 20                                           # strong signal: noisy ~ clean
    assert np.median(np.abs(ce.value[big] - cl.value[big]) / np.abs(ce.value[big])) < 0.2
