"""scripts/denoise_products.py (calibrated presence, calorimetry and clean maps) and the
full-map keys scripts/eval_denoise.py --products adds (no GPU, no data)."""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "..", "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DP = _load("denoise_products")
EV = _load("eval_denoise")
Q0 = DP.Q0_DEFAULT


def _prm(log_or=0.0, sigma2=0.0, tau=0.5, hurdle=False):
    return dict(q0=Q0, hurdle=hurdle, support_min_q=0.0, log_or=np.full((3, DP.ND), log_or).tolist(),
                sigma2=np.full((DP.NY, DP.ND), sigma2).tolist(), tau=[tau] * 3)


def test_box_excl_is_the_neighbourhood_without_the_cell():
    rng = np.random.default_rng(0)
    X = rng.random((40, 30))
    B = DP.box_excl(X, r=2)
    i, j = 10, 7
    assert np.isclose(B[i, j], X[i - 2:i + 3, j - 2:j + 3].sum() - X[i, j])
    assert np.isclose(B[0, 0], X[:3, :3].sum() - X[0, 0])                     # zero padding at the edge


def test_identity_without_corrections():
    y = np.array([0.0, 0.3, 1.0, 2.5]); p = np.array([0.1, 0.4, 0.8, 0.99])
    P = DP.products(y, p, 0, np.zeros(4), _prm())
    assert np.allclose(P["p_cal"], p)
    assert np.allclose(P["calo"], Q0 * np.expm1(y))                            # p_cal / p = 1, sigma^2 = 0
    assert np.allclose(P["clean"], np.where(p > 0.5, P["calo"], 0))
    H = DP.products(y, p, 0, np.zeros(4), _prm(hurdle=True))
    assert np.allclose(H["calo"], p * Q0 * np.expm1(y))                        # hurdle: E[q] = p * E[q | present]


def test_prior_shift_and_jensen():
    y = np.array([1.0]); p = np.array([0.9])
    P = DP.products(y, p, 1, np.array([5e4]), _prm(log_or=np.log(9.0), sigma2=0.04))
    assert np.isclose(P["p_cal"][0], 0.5)                                       # odds 9 / 9 = 1
    assert np.isclose(P["calo"][0], Q0 * np.expm1(1.02) * 0.5 / 0.9)           # Jensen + prior shift
    assert P["clean"][0] == 0.0                                                 # p_cal = 0.5 is not > tau = 0.5


def test_fm_keys_linearity_and_calibration():
    T = {0: np.zeros((32, 16)), 1: np.zeros((32, 16)), 2: np.zeros((32, 16))}
    M = {g: np.zeros((32, 16)) for g in T}
    cov = {g: np.ones((32, 16), bool) for g in T}
    T[0][3, 3], M[0][3, 3] = 2000.0, 1000.0                                     # under-predicted by 2
    M[0][10, 10] = 50.0                                                         # false charge (truth 0)
    S = EV._charge_stats(M, T, cov, Q0)
    k = EV.fm_keys(S, "charge")
    assert np.isclose(k["lin_cell_500-1k"], 2.0) and np.isclose(k["lin_cell_25-102"], 0.0)
    assert np.isclose(k["tot_ratio_U"], 1050 / 2000) and np.isclose(k["fm_false_frac"], 50 / 2000)
    assert np.isclose(k["fm_zeroed_frac"], 0.0) and np.isclose(k["fm_l1_frac"], 1050 / 2000)
    Pm = {g: np.full((32, 16), 0.25) for g in T}
    lab = {g: np.zeros((32, 16), bool) for g in T}
    for g in T:
        lab[g][:8] = True                                                       # base rate 0.25: calibrated
    kp = EV.fm_keys(EV._prob_stats(Pm, lab, cov), "prob")
    assert kp["cal_ece"] < 1e-12 and np.isclose(kp["cal_brier"], 0.1875)


def test_iso_keys_use_only_isolated_positives():
    rng = np.random.default_rng(1)
    W = []
    for p in range(3):
        for kind in ("bg", "bgn"):
            W += [dict(kind=kind, p=p, score=float(x), eb=-1) for x in rng.normal(0, 1, 1000)]
        W += [dict(kind="iso", p=p, score=10.0, eb=1, isolated=True) for _ in range(50)]
        W += [dict(kind="iso", p=p, score=-10.0, eb=1, isolated=False) for _ in range(50)]
    out = EV.iso_pp_scalars(W)
    assert out["floor_iso_pp_eff1pct_0.1-0.2"] == 1.0 and out["near_iso_pp_eff1pct_0.1-0.2_Y"] == 1.0
    assert EV.pp_scalars(W)["floor_pp_eff1pct_0.1-0.2"] == 0.5
