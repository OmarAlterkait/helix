"""helix.eval: reading a run, the evaluated events and their windows, the fine-tune's metric, the
along-wire probe, compare and the CLI."""
import pytest

torch = pytest.importorskip("torch")

import json
import os

import numpy as np

from helix.data import cells as C                                   # noqa: E402
from helix.model.checkpoint import load_run, run_dir, tokenizer_config  # noqa: E402
from helix.model.tokenize import PatchConfig, patch_config, tau_of_tick  # noqa: E402
from helix.data import cells as C
from helix.data.denoise import near_windows
from helix.eval import events as W


# ---- reading a run and the metrics -------------------------------------------------------

# helix.model.checkpoint.load_run and the helix.eval metrics, on CPU.
#
# The loader reads a run directory laid out the way pimm and helix's hooks write one (pimm's own
# split-checkpoint writer when pimm is importable); the metrics run on synthetic events whose
# truth is a cells record and whose tokens come from the training tokenizer. Pinned: the loader
# returns exactly the saved weights, the tokenizer of the train split and the step of the weight
# set it read, and refuses what it cannot read; the along-wire probe is seeded and learns what a
# feature carries (an informative "trained" arm beats geometry alone); a perfect denoiser scores
# perfectly, every probe event is scored, and an uncovered charged cell is a miss.

SMALL = dict(n_slot=8, n_band=4, n_plane=6, d=32, blocks=2, dec_blocks=1, heads=2, n_bins=16, gp=64, gd=128)
TOK = dict(cell_t="grid_center", pw=4, pt=2)
BL = np.array([216, 216, 432, 864, 1728])


def _config(data_root="/nowhere"):
    """What pimm's resolved_config.json holds for a helix run (the parts helix reads)."""
    return {"model": {"type": "Coeff-FM", "bins": "/archive/bins.pt", **SMALL},
            "save_path": "/somewhere",
            "data": {"train": {"type": "CoeffTPCDataset", "data_root": data_root, "split": ["run_x"],
                               "dataset_name": "sim_wire", "holdout": {"seed": 0, "fractions": {"train": 0.9, "probe": 0.1}},
                               "split_role": "train",
                               "transform": [{"type": "CoeffTokenize", "part": "coeff", "cfg": TOK},
                                             {"type": "CoeffCollect", "part": "coeff"}]}}}


def _model(seed):
    from helix.model import build_fm
    torch.manual_seed(seed)
    m = build_fm(dict(SMALL))
    m.set_bins(torch.linspace(-6, 6, SMALL["n_bins"] + 1).repeat(SMALL["n_band"], 1))
    return m


def _write_run(d, *, ema_step=5, last_step=7, prefix=""):
    """A run directory: resolved_config.json, provenance.json, an old run's EMA sidecar (the retired
    WeightEMA's layout), pimm's split checkpoint model/last (weights.pth + trainer.dcp) and an iter
    snapshot."""
    import torch.distributed.checkpoint as dcp
    d = str(d)
    os.makedirs(os.path.join(d, "model"), exist_ok=True)
    json.dump(_config(), open(os.path.join(d, "resolved_config.json"), "w"), default=str)
    json.dump([{"step": 0, "corpus": {"basis_digest": "abc"}, "helix": {"commit": "c0ffee", "dirty": False}}],
              open(os.path.join(d, "provenance.json"), "w"))
    raw, ema = _model(1), _model(2)
    sd = {prefix + k: v for k, v in raw.state_dict().items()}
    torch.save({"state_dict": {k: v.float() for k, v in ema.state_dict().items()}, "decay": 0.9999,
                "step": ema_step}, os.path.join(d, "model", "model_ema.pth"))
    last = os.path.join(d, "model", "last")
    try:
        from pimm.utils.checkpoints import save_split_checkpoint
        save_split_checkpoint({"model": {"state_dict": sd}, "trainer": {"global_step": last_step}}, last)
    except ImportError:
        os.makedirs(last)
        torch.save({"state_dict": sd}, os.path.join(last, "weights.pth"))
        dcp.save({"trainer": {"global_step": last_step, "epoch": 0}}, checkpoint_id=os.path.join(last, "trainer.dcp"))
    torch.save({"state_dict": sd}, os.path.join(d, "model", "iter_3.pth"))
    return raw, ema


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k].float(), sb[k].float()) for k in sa)


# ---------------------------------------------------------------------------------------- loader
def test_load_run_reads_the_last_checkpoint_and_an_old_runs_ema(tmp_path):
    raw, ema = _write_run(tmp_path)
    r = load_run(tmp_path, device="cpu")
    assert _same(r.model, raw) and not r.model.training
    assert not any(p.requires_grad for p in r.model.parameters())
    assert torch.isfinite(r.model.bin_edges).all()
    assert r.tokenizer == patch_config(TOK)
    p = r.provenance
    assert (p["weights"], p["step"], p["run"], p["bins"]) == ("last", 7, str(tmp_path), "/archive/bins.pt")
    assert p["corpus"]["identity"] == {"basis_digest": "abc"} and p["code"]["helix"]["commit"] == "c0ffee"
    r = load_run(tmp_path, "ema", device="cpu")                     # a run from before WeightEMA went
    assert _same(r.model, ema) and (r.provenance["weights"], r.provenance["step"]) == ("ema", 5)


def test_a_run_recording_retired_options_still_reads():
    """``cell_t`` (in TOK) and the training-only ``dec_frac`` / ``n_masks`` are in the resolved
    configs of runs made before they were removed: reading such a run ignores them."""
    from helix.model.checkpoint import model_config
    cfg = _config()
    cfg["model"].update(dec_frac=0.5, n_masks=2)
    assert not {"dec_frac", "n_masks", "type", "bins"} & set(model_config(cfg))
    assert "cell_t" in TOK and tokenizer_config(cfg) == PatchConfig(pw=4, pt=2)


def test_load_run_takes_one_checkpoint_and_finds_its_run(tmp_path):
    raw, _ = _write_run(tmp_path, prefix="module.")                  # a DDP-saved state_dict
    f = tmp_path / "model" / "iter_3.pth"
    assert run_dir(f) == str(tmp_path)
    r = load_run(f, device="cpu")
    assert _same(r.model, raw) and r.provenance["step"] == 3 and r.provenance["weights"] == "iter_3.pth"
    assert _same(load_run(tmp_path / "model" / "last", device="cpu").model, raw)


def test_load_run_refuses_what_it_cannot_read(tmp_path):
    _write_run(tmp_path)
    os.remove(tmp_path / "model" / "model_ema.pth")
    with pytest.raises(FileNotFoundError, match="model_ema.pth"):
        load_run(tmp_path, "ema", device="cpu")                      # no silent fallback to raw weights
    with pytest.raises(ValueError, match="weights="):
        load_run(tmp_path, "raw", device="cpu")
    deep = tmp_path / "a" / "b" / "c" / "d"                          # four levels below the run: not inside it
    deep.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="resolved_config.json"):
        run_dir(deep)
    sd = _model(0).state_dict()
    sd["bin_edges"] = torch.full_like(sd["bin_edges"], float("nan"))
    torch.save({"state_dict": sd}, tmp_path / "model" / "iter_9.pth")
    with pytest.raises(ValueError, match="bin_edges"):
        load_run(tmp_path / "model" / "iter_9.pth", device="cpu")


# ------------------------------------------------------------------------------ synthetic events
def _sample(rng, i, n_trk=60):
    """One event: compact particles on all six planes, each with a 3D point; coefficients under
    the charge in every tokenized band plus noise coefficients."""
    G, W, T, Q, TRK, YZ = [], [], [], [], [], []
    yz = rng.uniform(-2000, 2000, (n_trk, 2))
    for k in range(n_trk):
        for g in range(6):
            cw, ct = rng.integers(20, 900), rng.integers(100, 3000)
            ww, tt = np.meshgrid(np.arange(cw, cw + 6), np.arange(ct, ct + 40), indexing="ij")
            G.append(np.full(ww.size, g)); W.append(ww.ravel()); T.append(tt.ravel())
            Q.append(rng.uniform(10, 300, ww.size)); TRK.append(np.full(ww.size, k)); YZ.append(np.tile(yz[k], (ww.size, 1)))
    G, W, T, Q, TRK, YZ = (np.concatenate(x) for x in (G, W, T, Q, TRK, YZ))
    rec = C.cell_truth(G, W, T, Q, TRK, TRK, YZ)
    rec.update(trk_volume=(np.arange(n_trk) % 2).astype(np.int16), trk_id=np.arange(n_trk),
               trk_energy=rng.uniform(0.01, 2.0, n_trk), trk_motif=np.arange(n_trk) % 5, trk_pid=np.arange(n_trk) % 6)
    rec = {name: np.asarray(rec[name]).astype(dt) for _, _, name, dt in C.SCHEMA}
    cfg = patch_config(TOK)
    g, fw, ft = C.unkey(rec["key"])
    n = 3000                                       # noise: coefficients in every band, like charge
    g, fw, ft = (np.concatenate([a, b]) for a, b in zip((g, fw, ft), (rng.integers(0, 6, n), rng.integers(0, 480, n),
                                                                     rng.integers(5, 190, n))))
    rows = [(np.full(len(g), b), g, fw * C.FW, tau_of_tick(ft * C.FT + 8, g, np.full(len(g), b), BL, cfg))
            for b in range(4)]
    band, pg, wire, tau = (np.concatenate([r[j] for r in rows]) for j in range(4))
    key = np.unique(np.stack([band, pg, wire, tau], 1), axis=0)
    coeff = dict(band=key[:, 0].astype(np.uint8), plane_gid=key[:, 1].astype(np.int32), wire=key[:, 2].astype(np.int32),
                 tau=key[:, 3].astype(np.int32), value=rng.standard_normal((len(key), 1)).astype(np.float32) * 3,
                 _meta=dict(gids=np.arange(6), n_wires=np.full(6, 1000), band_lengths=BL,
                            norm_sigma=np.ones((6, 5), np.float32), sigma_norm=2.6))
    return dict(coeff=coeff, cells=rec, ident=dict(run="run_x", source_file="s.h5", event=i), name=f"e{i}")


class _DS(list):
    def get_data(self, i):
        return self[i]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    _write_run(d)
    rng = np.random.default_rng(0)
    return d, _DS(_sample(rng, i) for i in range(9))


def _events(run, ds):
    from helix.eval.events import events
    return list(events(ds, run.tokenizer, torch.device("cpu")))


def _pctx(run, n, n_boot=20, probe_seeds=2):
    from helix.eval.probe import ProbeContext, split_events
    return ProbeContext(run, run.model, split_events(n), 0, n_boot, probe_seeds)


def _dctx(run, denoiser, n_boot=20):
    from helix.eval import Context
    return Context(run, denoiser, 0, n_boot)


def _token_feature(ev, values, d=SMALL["d"], seed=0):
    """Per-token features whose column 0 is the mean of ``values`` over the cells a token covers
    in band 0 (the rest noise): what a representation that knows ``values`` would carry."""
    idx, _ = C.cell_inputs(ev.cells["key"], ev.cell_key, ev.band_lengths, patch_config(TOK))
    n = len(ev.cell_key)
    s, c = np.zeros(n), np.zeros(n)
    ok = idx[:, 0] >= 0
    np.add.at(s, idx[ok, 0], values[ok]); np.add.at(c, idx[ok, 0], 1)
    F = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    F[:, 0] = s / np.maximum(c, 1) * 3
    return torch.from_numpy(F)


def test_events_are_the_training_tokens_with_their_cells(synth):
    d, ds = synth
    run = load_run(d, device="cpu")
    evs = _events(run, ds)
    assert [e.index for e in evs] == list(range(len(ds)))
    e = evs[0]
    assert e.batch["inp"].shape == (len(e.cell_key), SMALL["n_slot"]) and e.batch["n_cells"] == len(e.cell_key)
    assert (np.diff(e.cell_key) > 0).all() and e.cells is ds[0]["cells"]
    f1 = e.features(run.model)
    assert f1 is e.features(run.model) and f1.shape == (len(e.cell_key), SMALL["d"])
    keys, amp = e.coeff_amp                          # norm_sigma 1: |value| itself, the largest per cell
    assert np.array_equal(keys, e.coeff_cells) and (amp > 0).all()
    k0 = C.coeff_cells(*(ds[0]["coeff"][x][:1] for x in ("band", "plane_gid", "wire", "tau")), run.tokenizer)
    assert amp[np.searchsorted(keys, k0[0])] >= abs(ds[0]["coeff"]["value"][0, 0]) - 1e-6


def test_the_raw_arm_on_raw_features_is_the_trained_arm(synth):
    """Fed the raw inputs as its 'encoder features', the trained arm IS the raw arm: same rows,
    same probe, same seed -- so trained - raw is what the features add over the input, exactly."""
    from helix.eval.probe import UProbe
    d, ds = synth
    run = load_run(d, device="cpu")
    m = UProbe(_pctx(run, len(ds), probe_seeds=1))
    for e in _events(run, ds):
        e._feats = e.raw_inputs()
        assert e._feats.shape == (len(e.cell_key), 2 * SMALL["n_slot"])
        m.add(e)
    up = m.result()
    assert up["trained"]["r2"] == up["raw"]["r2"] and up["d_over_raw"] == [0.0, 0.0, 0.0]
    assert up["trained"]["seed_half_range"]["r2"] == 0.0                         # one seed: no spread


def test_uprobe_learns_u_and_geometry_alone_does_not(synth):
    from helix.eval.probe import UProbe, u_of
    d, ds = synth
    run = load_run(d, device="cpu")
    m = UProbe(_pctx(run, len(ds)))
    for e in _events(run, ds):
        g = C.unkey(e.cells["key"])[0]
        e._feats = _token_feature(e, u_of(g, e.cells["dom_y"], e.cells["dom_z"]))
        m.add(e)
    out = m.result()
    assert out["trained"]["r2"][0] > 0.5 > out["geo"]["r2"][0]
    assert all(out["trained"][f"r2_{v}"][0] > 0.3 for v in "UVY")
    d, lo, hi = out["d_over_geo"]
    assert d == pytest.approx(out["trained"]["r2"][0] - out["geo"]["r2"][0], abs=1e-3) and lo <= d <= hi


def test_patch_rows_label_is_the_dominant_cells_charge_weighted_u():
    from helix.eval.probe import patch_rows, u_of
    cells = dict(key=C.fkey([0, 0, 0, 1], [1, 2, 3, 1], [5, 5, 5, 5]), q=np.array([1.0, 3.0, 5.0, 2.0]),
                 dom_frac=np.array([0.9, 0.8, 0.2, 1.0]), dom_y=np.array([0.0, 1000.0, 50.0, 0.0]),
                 dom_z=np.zeros(4))
    tok = np.array([[0, 1, -1, -1], [0, 1, -1, -1], [0, 1, -1, -1], [-1, -1, -1, -1]])   # one patch; one uncovered
    pr = patch_rows(cells, tok, 6)
    u = u_of([0, 0], [0.0, 1000.0], [0.0, 0.0])
    assert len(pr["y"]) == 1 and pr["plane"].tolist() == [0]
    assert pr["y"][0] == pytest.approx((1 * u[0] + 3 * u[1]) / 4)            # the non-dominant cell is out
    assert pr["geo"][0, :6].tolist() == [1, 0, 0, 0, 0, 0] and pr["geo"][0, -4:].tolist() == [1, 1, 0, 0]


def _oracle(scale=1.0, q_present=25.0, covered_only=True):
    """A denoiser that knows the truth: ``scale`` x each cell's charge (each covered cell's, with
    ``covered_only``), presence log-odds +20 above ``q_present`` e- and -20 below, the uncovered
    floor where no token covers the cell."""
    from helix.eval.denoise import UNCOVERED_LOGIT

    def predict(ev, keys):
        q = dict(zip(ev.cells["key"].tolist(), ev.cells["q"].tolist()))
        idx, _ = C.cell_inputs(keys, ev.cell_key, ev.band_lengths, patch_config(TOK))
        ok = (idx >= 0).any(1)
        t = np.array([q.get(k, 0.0) for k in keys.tolist()]) * (ok if covered_only else 1)
        logit = np.where(ok, np.where(t > q_present, 20.0, -20.0), UNCOVERED_LOGIT)
        return np.log1p(scale * t / C.Q0).astype(np.float32), logit.astype(np.float32)
    return predict


def test_a_perfect_denoiser_scores_perfectly_on_every_probe_event(synth):
    """No fit split: every event is scored, and a denoiser that knows the truth finds every clear
    and crowded window, its presence on identical windows with the amplitude's."""
    from helix.eval.denoise import Denoise
    d, ds = synth
    run = load_run(d, device="cpu")
    m = Denoise(_dctx(run, _oracle()))
    for e in _events(run, ds):
        m.add(e)
    out = m.result()
    win = out.pop("_windows")
    assert set(np.unique(win["meta/ev"]).tolist()) == set(range(len(ds)))       # all 9 events scored
    assert {"score/presence", "score/amplitude"} <= set(win) and len(win["score/presence"]) == len(win["meta/ev"])
    effs = [v[0] for k, v in out["presence"].items() if k.startswith("floor_eff1pct_") and v[0] == v[0]]
    assert effs and min(effs) == 1.0
    assert set(out) == {"presence", "amplitude", "cells"}


class _Fit:                                       # a fitted probe, as far as result() reads one
    val_score = 0.5


def _scored(cls, n_seeds=1, **state):
    """A probe metric in its scored state, set by hand (the rows a probe would have predicted)."""
    from helix.eval.probe import ProbeContext
    m = cls.__new__(cls)
    m.ctx = ProbeContext(None, None, 3, 0, 20, n_seeds)
    m.n_fit, m.n_patch = 10, 0
    for k, v in state.items():
        setattr(m, k, v)
    return m


def test_uprobe_ev_r2_weighs_every_group_equally():
    """One (event, plane) group of 200 rows predicted perfectly, three of 4 rows predicted
    backwards: the pooled R^2 follows the big group, the group-equal one the many small ones."""
    from helix.eval.probe import group_r2, plane_group
    from helix.eval.probe import UPROBE_ARMS, UProbe, seed_name
    rng = np.random.default_rng(0)
    sizes = [200, 4, 4, 4]
    grp = np.concatenate([np.full(s, plane_group(i, 0)) for i, s in enumerate(sizes)])
    y = rng.standard_normal(len(grp)).astype(np.float32)
    p = np.where(grp == plane_group(0, 0), y, -y)
    m = _scored(UProbe, probes={a: [_Fit()] for a in UPROBE_ARMS},
                p={seed_name(a, 0): [p] for a in UPROBE_ARMS}, y=[y], grp=[grp])
    out = m.result()["trained"]
    _, sse, sst = group_r2(p, y, grp)
    n = np.array(sizes)
    assert out["r2"][0] == pytest.approx(1 - sse.sum() / sst.sum(), abs=1e-4)
    assert out["r2_ev"][0] == pytest.approx(1 - (sse / n).sum() / (sst / n).sum(), abs=1e-4)
    assert out["r2"][0] > 0.5 > out["r2_ev"][0] and out["r2_ev_U"][0] == out["r2_ev"][0]


def test_the_denoise_charge_numbers_are_exact_for_the_truth_and_see_a_scale(synth):
    from helix.eval.denoise import Denoise, PRED_NAMES, TRUTH_NAMES
    d, ds = synth
    run = load_run(d, device="cpu")
    outs = {}
    for scale in (1.0, 1.1):
        m = Denoise(_dctx(run, _oracle(scale)))
        for e in _events(run, ds):
            m.add(e)
        outs[scale] = m.result()
    for scale, out in outs.items():
        cl = out["cells"]
        assert cl["n_cells"] > 0 and cl["presence_q"] == 25.0 and cl["frac_uncovered"] == 0   # synth: all covered
        for v in ("all", "U", "V", "Y"):
            bv = cl["by_view"][v]
            res = [r for r in bv["resolution"].values() if r["n"]]
            assert len(res) >= 2 and sum(r["n"] for r in bv["resolution"].values()) > 0
            for r in res:                                       # log(pred / true) = log(scale), no spread
                assert r["median"] == pytest.approx(np.log(scale), abs=1e-4) and r["hw68"] == pytest.approx(0, abs=1e-4)
                assert r["frac_zero"] == 0 and r["frac_uncovered"] == 0
            lin = [r for r in bv["linearity"].values() if r["n"]]
            assert lin and all(r["median_true_over_pred"] == pytest.approx(1 / scale, rel=1e-4) for r in lin)
            assert bv["closure"]["median"] == pytest.approx(scale, rel=1e-4) and bv["closure"]["n_events"] == len(ds)
            cal = cl["calibration"]["presence"][v]                   # presence is right either way
            assert cal["n"] > cl["by_view"][v]["resolution"][TRUTH_NAMES[0]]["n"] and 0 < cal["frac_true"] < 1
            assert cal["ece"] < 1e-6 and cal["brier"] < 1e-6 and cal["log_loss"] < 1e-6
        assert set(cl["by_view"]["all"]["linearity"]) == set(PRED_NAMES)
        assert "floor_eff1pct_crowded_near_0.1-0.2_mean" in out["presence"] and "near_fpr_at_far_Y" in out["amplitude"]


def test_an_uncovered_charged_cell_is_a_miss(synth):
    """With half of each event's tokens gone, the charged cells they covered are still scored --
    predicted 0, a miss -- even by a predictor that knows their charge: closure is the covered
    share of the charge, frac_uncovered the uncovered share of the cells. They used to be dropped,
    which scored a tokenizer that covers less as no worse."""
    from helix.eval.denoise import Denoise
    d, ds = synth
    run = load_run(d, device="cpu")
    evs = _events(run, ds)[:3]
    cov, q = [], []
    for e in evs:
        e.cell_key = e.cell_key[len(e.cell_key) // 2:]
        k, qq = np.asarray(e.cells["key"]), np.asarray(e.cells["q"], np.float64)
        k, qq = k[qq > 25], qq[qq > 25]
        cov.append((C.cell_inputs(k, e.cell_key, e.band_lengths, run.tokenizer)[0] >= 0).any(1)); q.append(qq)
    cov, q = np.concatenate(cov), np.concatenate(q)
    assert 0.05 < cov.mean() < 0.95
    m = Denoise(_dctx(run, _oracle(covered_only=False), n_boot=5))
    for e in evs:
        m.add(e)
    cl = m.result()["cells"]
    a = cl["by_view"]["all"]
    assert cl["n_cells"] == len(q) and cl["frac_uncovered"] == pytest.approx(1 - cov.mean(), abs=1e-6)
    assert a["frac_uncovered"] == pytest.approx(1 - cov.mean(), abs=1e-6)
    assert a["closure"]["pooled"] == pytest.approx(q[cov].sum() / q.sum(), rel=1e-4)
    for r in a["resolution"].values():
        if r["n"]:
            assert r["frac_zero"] == r["frac_uncovered"]          # the uncovered ones, and only they
    assert cl["calibration"]["presence"]["all"]["ece"] < 1e-6         # calibration: the covered cells


def test_binary_calibration():
    from helix.eval.denoise import binary_calibration
    rng = np.random.default_rng(0)
    y = np.arange(1000) < 700                                           # 70% true
    c = binary_calibration(np.full(1000, 0.7), y)
    assert c["ece"] == pytest.approx(0, abs=1e-9) and c["brier"] == pytest.approx(0.21) and c["frac_true"] == 0.7
    c = binary_calibration(np.full(1000, 0.9), y)                       # over-confident by 0.2
    assert c["ece"] == pytest.approx(0.2)
    assert c["log_loss"] == pytest.approx(-(0.7 * np.log(0.9) + 0.3 * np.log(0.1)), rel=1e-5)
    p = rng.random(200000)                                              # calibrated by construction
    assert binary_calibration(p, rng.random(200000) < p)["ece"] < 0.01


def test_identity_reads_the_digest_and_the_truth_cut(tmp_path):
    import h5py
    from types import SimpleNamespace
    from helix.eval.events import D1M_TRUTH_Q_CUT, identity
    paths = []
    for i, label in enumerate(({"hits_threshold": 25.0, "source": "hits"}, {"source": "hits"})):
        p = tmp_path / f"c{i}.h5"
        with h5py.File(p, "w") as f:
            f.create_group("config").attrs["label_json"] = json.dumps(label)
        paths.append(str(p))
    meta = {"a": {"basis_digest": b"d1g"}, "b": {"basis_digest": "d1g"}}        # bytes or str, as h5py gives
    ds = SimpleNamespace(coeff_reader=SimpleNamespace(_meta_by_path=meta),
                         cells_reader=SimpleNamespace(h5_files=paths[:1]))
    assert identity(ds) == dict(basis_digest="d1g", truth_q_cut=25.0,
                                truth_q_cut_source="cells /config label_json hits_threshold")
    ds.cells_reader.h5_files = paths[1:]                # a shard that records no cut: the default, said so
    got = identity(ds)
    assert got["truth_q_cut"] == D1M_TRUTH_Q_CUT and "default" in got["truth_q_cut_source"]
    assert identity([]) == dict(basis_digest=None, truth_q_cut=None, truth_q_cut_source=None)


def test_fit_probe_is_seeded_and_selects_on_the_metric_it_reports():
    from helix.eval.probe import fit_probe
    from helix.eval.probe import group_r2

    def r(p, y, g):
        _, sse, sst = group_r2(p, y, g)
        return float(1 - sse.sum() / sst.sum())
    rng = np.random.default_rng(0)
    n = 6000
    e = np.repeat(np.arange(12), n // 12)
    X = rng.standard_normal((n, 8)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.standard_normal(n) + 5 * (e % 3)).astype(np.float32)
    Xt, none = torch.from_numpy(X).half(), np.zeros((n, 0), np.float32)
    a = fit_probe(Xt, none, y, e, group=e, seed=1, device="cpu")
    b = fit_probe(Xt, none, y, e, group=e, seed=1, device="cpu")
    pa = a.predict(Xt, none)
    np.testing.assert_array_equal(pa, b.predict(Xt, none))
    assert r(pa, y, e) > 0.8 and a.val_score > 0.8          # R^2: charged for scale, not only rank
    # geometry alone: an unrelated column fits the between-event offsets, not the within-event signal
    geo = rng.standard_normal((n, 1)).astype(np.float32)
    g = fit_probe(torch.zeros((n, 0)), geo, y, e, group=e, device="cpu")
    assert r(g.predict(torch.zeros((n, 0)), geo), y, e) < 0.1
    # a prediction constant within every group explains nothing within them: R^2 0, not a NaN
    c = fit_probe(torch.zeros((n, 0)), (e % 3)[:, None].astype(np.float32), y, e, group=e, device="cpu")
    assert c.val_score == pytest.approx(0, abs=1e-6)


def test_fit_probe_uses_extra_columns_in_any_units_beside_many_features():
    """The geometry columns are standardised with the features: in their own units beside
    hundreds of unit-variance channels they were ignored (the along-wire random arm scored 0.005
    with columns that alone score 0.121)."""
    from helix.eval.probe import fit_probe
    rng = np.random.default_rng(0)
    n = 6000
    e = np.repeat(np.arange(12), n // 12)
    s = rng.standard_normal(n).astype(np.float32)
    X = torch.from_numpy(rng.standard_normal((n, 512)).astype(np.float32)).half()      # noise
    A = np.stack([1e-3 * s, np.ones(n, np.float32)], 1)               # the signal, tiny units; a constant
    p = fit_probe(X, A, s, e, seed=0, device="cpu").predict(X, A)
    assert np.corrcoef(p, s)[0, 1] > 0.9


def test_the_probe_fits_on_a_budget_and_keeps_no_features_of_a_scored_event(synth, monkeypatch):
    import helix.eval.probe as P
    rng = lambda: np.random.default_rng(3)                                             # noqa: E731
    assert np.array_equal(P.fit_rows(5, 4, rng()), np.arange(5))
    monkeypatch.setattr(P, "FIT_ROWS", 3000)         # 500 rows per training event: binds
    k = P.fit_rows(1000, 7, rng())
    assert len(k) == 428 and np.array_equal(k, np.unique(k)) and np.array_equal(k, P.fit_rows(1000, 7, rng()))
    d, ds = synth
    run = load_run(d, device="cpu")
    ctx = _pctx(run, len(ds), n_boot=5)
    m = P.UProbe(ctx)
    for e in _events(run, ds):
        m.add(e)
        assert (m.fit is None) == (e.index >= ctx.n_train)      # fitted at the first scored event
    assert P.gather(torch.zeros((5, 3)), np.zeros((0, 4), np.int64)).shape == (0, 12)   # an event without rows
    held = list(m.p["trained"])
    assert held and all(isinstance(x, np.ndarray) and x.ndim == 1 for x in held)    # predictions, no features
    assert 0 < m.result()["n_fit_rows"] <= 3000


def _fake_corpus(ds, seen=None):
    """``helix.eval._corpus`` on synthetic events: the dataset and what resolution would record."""
    def corpus(run, override):
        if seen is not None:
            seen["override"] = override
        return ds, dict(recorded="/nowhere", resolved=override or "/nowhere", check="corpus identity NOT "
                        "RECORDED by this run (predates the stamp); reading basis_digest=... unchecked")
    return corpus


def test_the_cli_scores_a_fine_tuned_denoiser_run(synth, tmp_path, monkeypatch):
    """``python -m helix.eval DENOISER_RUN``: the run's ``last`` weights, its FM run where recorded,
    the corpus as resolved, one JSON and the two window scores beside it -- which compare pairs
    within the run (presence against amplitude)."""
    import helix.eval as E
    from helix.eval.__main__ import main
    from test_denoise import _write_denoise_run
    d, ds = synth
    den = tmp_path / "den"
    den.mkdir()
    _write_denoise_run(str(den), d)
    seen = {}
    monkeypatch.setattr(E, "_corpus", _fake_corpus(ds, seen))
    out = tmp_path / "e.json"
    assert main([str(den), "--out", str(out), "--max-events", "3", "--boot", "5", "--corpus", "/here"]) == 0
    res = json.load(open(out))
    assert res["n_events"] == 3 and set(res["metrics"]) == {"denoise"} and seen["override"] == "/here"
    assert res["run"]["weights"] == "last" and res["run"]["checkpoint"].endswith(os.path.join("last", "weights.pth"))
    assert res["fm"] == dict(recorded=str(d), resolved=str(d), note="as recorded")
    assert res["corpus"]["resolved"] == "/here" and "unchecked" in res["corpus"]["check"]
    assert set(res["truth"]) == {"q_cut", "q_cut_source"} and "_windows" not in res["metrics"]["denoise"]
    assert set(res["metrics"]["denoise"]) == {"presence", "amplitude", "cells"}
    z = np.load(tmp_path / "e.windows.npz")
    assert {f"meta/{k}" for k in W.WINDOW_META} | {"score/presence", "score/amplitude"} == set(z.files)
    assert set(np.unique(z["meta/ev"]).tolist()) == {0, 1, 2}
    pa = tmp_path / "pa.json"
    assert main(["compare", "--a", str(tmp_path / "e.windows.npz"), "--arm", "presence", "--arm-b", "amplitude",
                 "--boot", "5", "--out", str(pa)]) == 0
    r = json.load(open(pa))
    assert r["arm_b"] == "amplitude" and "floor_eff1pct_clear_0.1-0.2_mean" in r["a_minus_b"]


def test_the_cli_runs_the_along_wire_probe_on_an_fm_run(synth, tmp_path, monkeypatch):
    import helix.eval as E
    from helix.eval.__main__ import main
    d, ds = synth
    monkeypatch.setattr(E, "_corpus", _fake_corpus(ds))
    out = tmp_path / "u.json"
    assert main(["--uprobe", str(d), "--out", str(out), "--probe-seeds", "1", "--boot", "5"]) == 0
    res = json.load(open(out))
    assert set(res["metrics"]) == {"uprobe"} and res["run"]["weights"] == "last" and res["probe_seeds"] == 1
    assert res["n_events"] == len(ds) and res["metrics"]["uprobe"]["n_train_events"] == 6
    assert "r2" in res["metrics"]["uprobe"]["trained"] and not (tmp_path / "u.windows.npz").exists()
    with pytest.raises(SystemExit, match="--fm"):
        main(["--uprobe", str(d), "--fm", str(d), "--out", str(out)])


def test_resolve_exp_relocates_another_sites_run(tmp_path, monkeypatch):
    """A denoiser trained at S3DF names its FM run by an S3DF path; here it is the same path under
    this site's HELIX_EXP, or what ``--fm`` says. Neither: refused, naming the cure."""
    from helix.eval import resolve_exp
    rec = "/sdf/data/neutrino/omara/exp/helix/fm_cooldown/model/last"   # helix/sites/s3df.yaml's HELIX_EXP
    if os.path.exists(rec):
        pytest.skip("at S3DF: the recorded path exists")
    monkeypatch.setenv("HELIX_EXP", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="--fm"):
        resolve_exp(rec)
    (tmp_path / "fm_cooldown" / "model" / "last").mkdir(parents=True)
    path, note = resolve_exp(rec)
    assert path == str(tmp_path / "fm_cooldown" / "model" / "last") and "s3df" in note
    assert resolve_exp(str(tmp_path), override=str(tmp_path / "fm_cooldown")) == (str(tmp_path / "fm_cooldown"), "given")
    assert resolve_exp(str(tmp_path))[1] == "as recorded"
    with pytest.raises(FileNotFoundError, match="--fm"):
        resolve_exp(rec, override=str(tmp_path / "nope"))


def test_the_ft_eval_config_is_the_protocol_and_needs_four_ranks(monkeypatch):
    """configs/pimm/ft_eval.py: 1,024 labelled events for 16 epochs at one event per rank is 4,096
    steps only on 4 ranks, so any other world size is refused rather than run at another length."""
    import importlib.util
    if importlib.util.find_spec("pimm") is None:
        pytest.skip("pimm not importable in this env")
    from pimm.utils.config import Config
    from helix.paths import SiteError
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "pimm", "ft_eval.py")
    monkeypatch.setenv("WORLD_SIZE", "4")
    try:
        cfg = Config.fromfile(path)
    except (SiteError, FileNotFoundError, SystemExit) as exc:
        pytest.skip(f"config needs a root this site does not configure: {exc}")
    t, m = cfg.data["train"], cfg.model
    assert (t["n_events"], t["subset_seed"], cfg.epoch) == (1024, 0, 16) and cfg.batch_size == 4
    assert m["type"] == "Coeff-Denoise" and m["cond_charge"] and m["dist"] == "point" and m["lin_amp_scale"] == 0.0
    assert m["pretrained"] and not m["freeze_encoder"]                  # the frozen diagnostic is an override
    for ws in ("2", "16"):
        monkeypatch.setenv("WORLD_SIZE", ws)
        with pytest.raises(Exception, match="WORLD_SIZE"):
            Config.fromfile(path)


# ---- the windows and the fine-grid join --------------------------------------------------

# The evaluation windows (helix.eval.events) and the fine-grid join in helix.data.cells.
#
# Synthetic events, CPU, numpy only. Each test pins one property the metrics must have for a
# patch-size comparison to mean anything: keys round-trip, a perfect prediction finds every
# isolated particle, the false-positive rate is set per view, windows are not centred on the
# particle they score, the within-group R^2 ignores offsets and charges scale, and a cell finds
# the tokens covering it with the tokenizer's own arithmetic.

def test_keys_round_trip_including_negative_ticks():
    g, fw, ft = np.array([0, 5, 3]), np.array([0, 17, 984]), np.array([-30, 0, 270])
    assert all((a == b).all() for a, b in zip(C.unkey(C.fkey(g, fw, ft)), (g, fw, ft)))


def _event(rng, n_part=40):
    """Compact particles (a few wires x a few ticks) scattered over one plane, as a cells record."""
    g, w, t, q, trk = [], [], [], [], []
    for i in range(n_part):
        cw, ct = rng.integers(40, 900), rng.integers(200, 3000)
        ww, tt = np.meshgrid(np.arange(cw, cw + 3), np.arange(ct, ct + 20), indexing="ij")
        g.append(np.zeros(ww.size, int)); w.append(ww.ravel()); t.append(tt.ravel())
        q.append(np.full(ww.size, 100.0)); trk.append(np.full(ww.size, i))
    g, w, t, q, trk = map(np.concatenate, (g, w, t, q, trk))
    rec = C.cell_truth(g, w, t, q, trk, trk, np.zeros((len(q), 2)))
    rec["trk_energy"] = 0.05 * (np.arange(n_part) + 1)
    cells = C.fkey(g, w // C.FW, t // C.FT)
    noise = C.fkey(np.zeros(500, int), rng.integers(0, 500, 500), rng.integers(0, 250, 500))
    return rec, np.concatenate([cells, noise])


N_CELL = (W.WW // C.FW) * (W.WT // C.FT)


def _charge(rec, keys):
    k, q = np.asarray(rec["key"]), np.asarray(rec["q"], np.float64)
    p = np.clip(np.searchsorted(k, keys), 0, len(k) - 1)
    return np.where(k[p] == keys, q[p], 0.0)


def _haze(rec, keys, value, r=8):
    """``value`` on every cell of ``keys`` with truth charge within ``r`` cells (wire and tick, its
    plane), else 0: what a prediction that blurs charge into the surroundings of tracks adds."""
    g, fw, ft = C.unkey(np.asarray(rec["key"])[np.asarray(rec["q"]) > 0])
    kg, kw, kt = C.unkey(keys)
    out = np.zeros(len(keys))
    for p in np.unique(kg):
        on = g == p
        occ = np.zeros((int(max(fw.max(initial=0), kw.max())) + 2 * r + 2,
                        int(max(ft.max(initial=0), kt.max())) + 2 * r + 2))
        occ[fw[on] + r, ft[on] + r] = 1
        S = np.pad(occ.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        m = kg == p
        i0, i1, j0, j1 = kw[m], kw[m] + 2 * r + 1, kt[m], kt[m] + 2 * r + 1        # grid offset r
        out[m] = np.where(S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0] > 0, value, 0.0)
    return out


def test_truth_has_both_window_kinds_and_valid_keys():
    rec, ck = _event(np.random.default_rng(0))
    tr = W.event_truth(rec, ck, np.random.default_rng(1))
    kinds = [m["kind"] for m in tr["meta"]]
    assert {"iso", "bg", "near"} <= set(kinds) <= {"iso", "bg", "near", "isox"}
    assert kinds.count("bg") <= W.N_BG_PLANE * 1 and kinds.count("near") <= W.N_BG_PLANE    # one plane here
    assert len(tr["wkey"]) == len(tr["meta"]) * N_CELL
    q = _charge(rec, tr["wkey"])
    near_charge = _haze(rec, tr["wkey"], 1.0) > 0                 # charge within 8 cells
    for i, m in enumerate(tr["meta"]):
        cells = slice(i * N_CELL, (i + 1) * N_CELL)
        assert np.array_equal(tr["wkey"][cells], W.window_cells(m["g"], m["w0"], m["t0"]))
        if m["kind"] == "bg":
            assert q[cells].sum() == 0 and np.isnan(m["E"]) and not near_charge[cells].any()
            assert np.isin(tr["wkey"][cells], ck).any()
        elif m["kind"] == "near":                                  # empty, a kept coefficient, charge beside
            assert q[cells].sum() == 0 and np.isnan(m["E"]) and near_charge[cells].any()
            assert np.isin(tr["wkey"][cells], ck).any()
        else:
            assert m["E"] > 0 and q[cells].sum() > 0
            if m["kind"] == "isox":                                # the extra draws are clear ones
                assert m["ctx"] == "clear"
    # near, then the extra clear windows, are drawn last: iso and bg are what they were without them
    from helix.eval import events as EV
    import unittest.mock as um
    with um.patch("helix.data.denoise.near_windows", lambda *a, **k: []):
        old = EV.event_truth(rec, ck, np.random.default_rng(1))
    keep = lambda meta: [m for m in meta if m["kind"] in ("iso", "bg")]          # noqa: E731
    assert keep(old["meta"]) == keep(tr["meta"])


def test_every_plane_gets_its_own_background():
    rec, ck = _event(np.random.default_rng(0))
    g, fw, ft = C.unkey(ck)
    ck = np.concatenate([ck, C.fkey(np.full(len(ck), 2), fw, ft)])      # a second plane: noise only
    tr = W.event_truth(rec, ck, np.random.default_rng(1))
    bg = [m["g"] for m in tr["meta"] if m["kind"] == "bg"]
    assert bg.count(0) == W.N_BG_PLANE and bg.count(2) == W.N_BG_PLANE


def test_far_background_is_uniform_in_area_not_in_kept_count():
    """A noise-only plane with one window-sized patch fully kept (128 cells) and one kept cell per
    window-sized block elsewhere (100): a far window lands on the patch at about its share of the
    AREA (a few %), not its share of the kept cells (128/228, which centring on a kept cell gave)."""
    rec, ck = _event(np.random.default_rng(0))
    nw, nt = W.WW // C.FW, W.WT // C.FT
    bw, bt = np.meshgrid(np.arange(10) * nw, np.arange(10) * nt, indexing="ij")
    sparse = C.fkey(np.full(100, 3), bw.ravel() + 3, bt.ravel() + 2)
    pw, pt = np.meshgrid(np.arange(nw), np.arange(nt), indexing="ij")
    patch = C.fkey(np.full(nw * nt, 3), pw.ravel() + 5 * nw, pt.ravel() + 5 * nt)
    ck3 = np.unique(np.concatenate([ck, sparse, patch]))
    on = 0
    n = 0
    for s in range(20):
        tr = W.event_truth(rec, ck3, np.random.default_rng(s))
        for i, m in enumerate(tr["meta"]):
            if m["kind"] == "bg" and m["g"] == 3:
                n += 1
                on += np.isin(tr["wkey"][i * N_CELL:(i + 1) * N_CELL], patch).any()
    assert n == 20 * W.N_BG_PLANE and on / n < 0.15


def test_windows_are_not_centred_on_the_particle():
    rec, ck = _event(np.random.default_rng(0))
    tr = W.event_truth(rec, ck, np.random.default_rng(1))
    q = _charge(rec, tr["wkey"]).reshape(len(tr["meta"]), W.WW // C.FW, W.WT // C.FT)
    wire = np.arange(W.WW // C.FW)[:, None]
    off = [(qi * wire).sum() / qi.sum() - (W.WW // C.FW - 1) / 2 for qi, m in zip(q, tr["meta"]) if m["kind"] == "iso"]
    assert np.std(off) > 0.5


def test_a_perfect_prediction_scores_perfectly():
    rec, ck = _event(np.random.default_rng(0))
    tr = W.event_truth(rec, ck, np.random.default_rng(1))
    out = W.score_windows(_charge(rec, tr["wkey"]), [dict(m, ev=0) for m in tr["meta"]], n_boot=5)
    effs = [v[0] for k, v in out.items() if k.startswith("floor_eff1pct_") and v[0] == v[0]]
    assert effs and all(v == 1.0 for v in effs)        # every isolated particle beats empty windows
    assert out["n_windows"]["bg_U"] > 0 and out["n_windows"]["bg_V"] == 0
    assert out["near_fpr_at_far_U"][0] == 0.0 and out["n_windows"]["near_U"] > 0
    assert np.isnan(out["floor_eff1pct_clear_0.1-0.2_V"][0])            # no V window: not a 0
    assert np.isnan(out["floor_eff1pct_clear_0.1-0.2_mean"][0])         # nor is the view mean
    assert not any(k.startswith("auc") or k.endswith("_ev") for k in out)


def test_a_haze_around_tracks_is_free_on_the_far_background_and_priced_on_the_near():
    """A prediction that puts charge (above any particle's) on every cell within 8 cells of truth
    detects every clear particle against the far background, which keeps those surroundings
    empty by construction -- and no crowded one against the near background, on which the far
    threshold fires every time."""
    rec, ck = _event(np.random.default_rng(0), n_part=120)
    tr = W.event_truth(rec, ck, np.random.default_rng(1))
    meta = [dict(m, ev=0) for m in tr["meta"]]
    q = _charge(rec, tr["wkey"])
    hazy = np.maximum(q, _haze(rec, tr["wkey"], 1.5 * q.max()))
    good, bad = (W.score_windows(s, meta, n_boot=5) for s in (q, hazy))
    ok = lambda k: good[k][0] == good[k][0]                                       # noqa: E731
    clear = [nm for nm in W.ENERGY_NAMES if ok(f"floor_eff1pct_clear_{nm}")]
    crowded = [nm for nm in W.ENERGY_NAMES if ok(f"floor_eff1pct_crowded_near_{nm}")]
    assert clear and crowded
    for nm in clear:
        assert bad[f"floor_eff1pct_clear_{nm}"][0] == 1.0 == good[f"floor_eff1pct_clear_{nm}"][0]   # far: blind
        assert bad[f"floor_eff1pct_clear_{nm}_U"][0] == 1.0
    for nm in crowded:
        assert bad[f"floor_eff1pct_crowded_near_{nm}"][0] == 0.0 and good[f"floor_eff1pct_crowded_near_{nm}"][0] == 1.0
        assert bad[f"floor_eff1pct_crowded_near_{nm}_U"][0] == 0.0
    assert bad["near_fpr_at_far_U"][0] == 1.0 and good["near_fpr_at_far_U"][0] == 0.0


def test_the_matched_floor_compares_iso_and_background_at_the_same_distance_from_charge():
    """A predictor that hazes charge near activity: far windows score 0, near windows 3. Iso
    windows with clear padding score 5 and are all found at the far threshold; crowded iso
    windows score like the near windows (3, the haze alone), so the matched near threshold finds
    none of them. Mixed iso windows are in no number."""
    meta, score = [], []
    for e in range(10):
        meta += [dict(kind="bg", g=0, E=np.nan, ev=e)] * 20 + [dict(kind="near", g=0, E=np.nan, ev=e)] * 20
        score += [0.0] * 20 + [3.0] * 20
        meta += [dict(kind="iso", g=0, E=0.15, ev=e, ctx="clear")] * 3
        meta += [dict(kind="iso", g=0, E=0.15, ev=e, ctx="crowded")] * 3
        meta += [dict(kind="iso", g=0, E=0.15, ev=e, ctx="mixed")] * 3
        score += [5.0] * 3 + [3.0] * 3 + [9.0] * 3
    stat, _, counts = W.window_stat(np.repeat(np.asarray(score), N_CELL), meta)
    got = stat(np.ones(10, np.int64))
    assert got["floor_eff1pct_clear_0.1-0.2"] == 1.0 == got["floor_eff1pct_clear_0.1-0.2_U"]
    assert got["floor_eff1pct_crowded_near_0.1-0.2"] == 0.0 == got["floor_eff1pct_crowded_near_0.1-0.2_U"]
    assert np.isnan(got["floor_eff1pct_clear_0.1-0.2_mean"])           # V and Y have no windows
    assert counts["clear_0.1-0.2_U"] == 30 == counts["crowded_0.1-0.2_U"] and counts["iso_U"] == 90
    assert got["near_fpr_at_far_U"] == 1.0 and np.isnan(got["near_fpr_at_far_V"])
    assert set(got) == ({f"floor_eff1pct_{k}_{nm}{v}" for k in ("clear", "crowded_near") for nm in W.ENERGY_NAMES
                         for v in ("", "_U", "_V", "_Y", "_mean")} | {f"near_fpr_at_far_{v}" for v in W.VIEWS})


def test_the_view_mean_is_the_mean_of_the_three_views():
    """Clear windows on all three views, found on U only: per view 1, 0, 0; their mean 1/3; the
    pooled value weighs the views by their window counts (U has 3 of the 5 per event)."""
    meta, score = [], []
    for e in range(10):
        for g, n_iso, s in ((0, 3, 5.0), (1, 1, 0.0), (2, 1, 0.0)):
            meta += [dict(kind="bg", g=g, E=np.nan, ev=e)] * 20 + [dict(kind="iso", g=g, E=0.05, ev=e, ctx="clear")] * n_iso
            score += [1.0] * 20 + [s] * n_iso
    got = W.window_stat(np.repeat(np.asarray(score), N_CELL), meta)[0](np.ones(10, np.int64))
    k = "floor_eff1pct_clear_0-0.1"
    assert [got[f"{k}_{v}"] for v in W.VIEWS] == [1.0, 0.0, 0.0]
    assert got[f"{k}_mean"] == pytest.approx(1 / 3) and got[k] == pytest.approx(3 / 5)


def test_the_extra_clear_windows_count_in_clear():
    """isox windows (extra clear draws) score in the clear efficiency; crowded ones are iso alone."""
    meta, score = [], []
    for e in range(10):
        meta += [dict(kind="bg", g=0, E=np.nan, ev=e)] * 20
        score += [0.0] * 20
        meta += [dict(kind="iso", g=0, E=0.15, ev=e, ctx="crowded"), dict(kind="isox", g=0, E=0.15, ev=e, ctx="clear")]
        score += [0.0, 5.0]
    stat, _, counts = W.window_stat(np.repeat(np.asarray(score), N_CELL), meta)
    got = stat(np.ones(10, np.int64))
    assert got["floor_eff1pct_clear_0.1-0.2"] == 1.0 and np.isnan(got["floor_eff1pct_crowded_near_0.1-0.2"])
    assert counts["isox_U"] == 10 and counts["clear_0.1-0.2_U"] == 10 and counts["iso_U"] == 10


def test_the_threshold_is_each_views_own():
    """A noisy view's background sets only that view's threshold: a quiet view's particles,
    above its own background but below the noisy one's, are all detected."""
    rng = np.random.default_rng(0)
    meta, score = [], []
    for g, scale in ((0, 1.0), (1, 1.0), (2, 10.0)):
        for _ in range(300):
            meta.append(dict(kind="bg", g=g, E=np.nan, ev=int(rng.integers(0, 30))))
            score.append(scale * rng.random())
        for _ in range(50):
            meta.append(dict(kind="iso", g=g, E=0.15, ev=int(rng.integers(0, 30)), ctx="clear"))
            score.append(scale * (1.5 + rng.random()))
    out = W.score_windows(np.repeat(score, N_CELL), meta, n_boot=5)
    assert out["floor_eff1pct_clear_0.1-0.2"][0] == 1.0
    pooled = np.quantile([s for s, m in zip(score, meta) if m["kind"] == "bg"], 0.99)
    assert np.mean([s > pooled for s, m in zip(score, meta) if m["kind"] == "iso"]) < 0.7   # what one pooled cut gives


def test_group_r2_ignores_offsets_and_charges_scale():
    from helix.eval.probe import group_r2
    rng = np.random.default_rng(0)
    grp = np.repeat(np.arange(10), 50)
    y = rng.standard_normal(500) + 10 * grp                 # between-group variance dominates
    r2 = lambda p: (lambda k, sse, sst: 1 - sse.sum() / sst.sum())(*group_r2(p, y, grp))   # noqa: E731
    assert r2(y + 7 * rng.standard_normal(10)[grp]) == pytest.approx(1.0)      # a per-group offset is free
    assert r2(np.zeros(500)) == pytest.approx(0.0) and r2(10 * grp) == pytest.approx(0.0)
    assert r2(3 * y) < 0                                 # perfectly correlated, wrong scale
    keys, sse, sst = group_r2(y, y, grp)
    assert keys.tolist() == list(range(10)) and (sse == 0).all() and (sst > 0).all()


def test_bootstrap_weighs_an_event_by_how_often_it_was_drawn():
    v, seen = np.arange(20.0), []

    def stat(m):
        seen.append(m)
        return {"m": float((m * v).sum() / m.sum())}

    pt, lo, hi = W.bootstrap(stat, 20, n_boot=200)["m"]
    assert pt == pytest.approx(9.5) and lo < pt < hi
    assert (seen[0] == 1).all() and all(m.sum() == 20 for m in seen) and any(m.max() > 1 for m in seen)


def test_cell_inputs_matches_the_gather_arithmetic():
    """cell_inputs: the covering tokens and offset features of the tokenizer's own pixel_cells."""
    from helix.model.tokenize import PatchConfig, pixel_cells
    cfg = PatchConfig()
    rng = np.random.default_rng(0)
    bl = np.array([280, 280, 560, 1120])
    keys = C.fkey(rng.integers(0, 6, 400), rng.integers(0, 900, 400), rng.integers(0, 260, 400))
    g, fw, ft = C.unkey(keys)
    w, t = fw * C.FW, ft * C.FT + C.FT // 2
    pc = pixel_cells(g, w, t, bl, cfg)
    tok_keys = np.unique(pc[rng.random(pc.shape) < 0.5])        # a token set covering about half
    idx, aux = C.cell_inputs(keys, tok_keys, bl, cfg)
    order = np.argsort(tok_keys); cks = tok_keys[order]
    pos = np.clip(np.searchsorted(cks, pc), 0, len(cks) - 1)
    ref_idx = np.where(cks[pos] == pc, order[pos], -1)
    dec = (1 << np.asarray(cfg.lev)).astype(np.float64); toff = np.asarray(cfg.toff)[g % 3]
    offs = []
    for b in range(cfg.n_bands):
        tau = (t + toff) / dec[b] - cfg.delta[b]
        offs += [(w % cfg.pw + 0.5) / cfg.pw, (tau / cfg.pt) % 1.0]
    x = np.stack(offs, 1).astype(np.float32)
    ang = 2 * np.pi * x[..., None] * np.array([1, 2, 3], np.float32)
    ref_aux = np.concatenate([(ref_idx >= 0).astype(np.float32),
                              np.concatenate([np.sin(ang), np.cos(ang)], -1).reshape(len(keys), -1)], 1)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(aux, ref_aux)
    assert aux.shape == (len(keys), 13 * cfg.n_bands)


def test_coeff_cells_are_the_cells_under_the_kept_coefficients():
    from helix.model.tokenize import PatchConfig, tick_of_tau
    cfg = PatchConfig()
    band, g, wire, tau = np.array([0, 3, 4, 0]), np.array([1, 2, 2, 1]), np.array([10, 11, 12, 10]), np.array([5, 9, 9, 5])
    got = C.coeff_cells(band, g, wire, tau, cfg)
    k = band < cfg.n_bands                                              # D1 is not tokenized
    want = np.unique(C.fkey(g[k], wire[k] // C.FW, np.floor(tick_of_tau(tau[k], g[k], band[k], cfg) / C.FT)))
    np.testing.assert_array_equal(got, want)
    assert len(got) == 2


def test_near_windows_are_empty_but_beside_charge():
    rng = np.random.default_rng(3)
    keys, q = [], []
    for g in (0, 4):                                               # two planes, a few compact blobs each
        for _ in range(6):
            cw, ct = rng.integers(20, 400), rng.integers(20, 200)
            ww, tt = np.meshgrid(np.arange(cw, cw + 3), np.arange(ct, ct + 2), indexing="ij")
            keys.append(C.fkey(np.full(ww.size, g), ww.ravel(), tt.ravel())); q.append(np.full(ww.size, 500.0))
    keys, q = np.concatenate(keys), np.concatenate(q)
    order = np.argsort(keys); keys, q = keys[order], q[order]
    wins = near_windows(keys, q, np.random.default_rng(0), n=20)
    assert 0 < len(wins) <= 20 and {g for g, _, _ in wins} == {0, 4}
    G, CW, CT = C.unkey(keys)
    for g, w0, t0 in wins:
        assert w0 % C.FW == 0 and t0 % C.FT == 0
        fw, ft = w0 // C.FW, t0 // C.FT
        on = G == g
        inside = (CW >= fw - 1) & (CW < fw + W.WW // C.FW + 1) & (CT >= ft - 1) & (CT < ft + W.WT // C.FT + 1)
        near = (CW >= fw - 8) & (CW < fw + W.WW // C.FW + 8) & (CT >= ft - 8) & (CT < ft + W.WT // C.FT + 8)
        assert not (on & inside).any() and (on & near).any()
    assert wins == near_windows(keys, q, np.random.default_rng(0), n=20)


def test_cell_neighbors_centre_is_the_covering_token_and_shifts_by_one_patch():
    from helix.model.tokenize import PatchConfig, cell_key, pixel_cells, unpack_cell_key
    cfg = PatchConfig()
    bl = np.array([271, 271, 542, 1084, 2168])
    rng = np.random.default_rng(0)
    toks = []
    for b in range(cfg.n_bands):
        for wb in range(2, 8):
            for tb in range(2, 12):
                toks.append(cell_key(1, b, wb * cfg.pw, tb * cfg.pt, cfg))
    ck = np.array(rng.permutation(toks), np.int64)
    keys = C.fkey(np.full(40, 1), rng.integers(40, 100, 40), rng.integers(20, 60, 40))
    nbr = C.cell_neighbors(keys, ck, bl, cfg)
    idx, _ = C.cell_inputs(keys, ck, bl, cfg)
    assert nbr.shape == (40, cfg.n_bands, 9) and np.array_equal(nbr[:, :, 4], idx)
    g, fw, ft = C.unkey(keys)
    pc = pixel_cells(g, fw * C.FW, ft * C.FT + C.FT // 2, bl, cfg)
    for j, (dw, dt) in enumerate([(a, c) for a in (-1, 0, 1) for c in (-1, 0, 1)]):
        hit = nbr[:, :, j] >= 0
        _, _, wb, tb = unpack_cell_key(ck[nbr[:, :, j][hit]])
        _, _, wb0, tb0 = unpack_cell_key(pc[hit])
        assert np.array_equal(wb, wb0 + dw) and np.array_equal(tb, tb0 + dt)


def _dump(path, meta, **scores):
    """A windows dump as the CLI writes one: ``meta/<key>`` per window, ``score/<name>``."""
    arr = lambda k: np.array([m[k] for m in meta])                              # noqa: E731
    np.savez(path, **{f"meta/{k}": arr(k) for k in meta[0]}, **{f"score/{k}": v for k, v in scores.items()})


def _clear_meta(n_ev=20, views=(0,), n_bg=30, n_iso=5):
    meta = []
    for e in range(n_ev):
        for g in views:
            meta += [dict(ev=e, kind="bg", g=g, E=np.nan, ctx="", w0=0, t0=0)] * n_bg
            meta += [dict(ev=e, kind="iso", g=g, E=0.15, ctx="clear", w0=32, t0=0)] * n_iso
    return meta


def test_compare_is_paired_over_identical_windows(tmp_path):
    """Two runs scored on the same windows: A - B over shared resamples, zero for identical
    scores and exact for a known difference; different windows -- events, origins, a key only
    one dump has -- are refused."""
    from helix.eval.__main__ import main as cli
    rng = np.random.default_rng(0)
    meta = _clear_meta()
    s_bg = rng.random(len(meta))
    is_iso = np.array([m["kind"] == "iso" for m in meta])
    a = np.where(is_iso, 2.0, s_bg)                      # every iso window above every bg: found
    b = np.where(is_iso, np.where(np.arange(len(meta)) % 2 == 0, 2.0, 0.0), s_bg)   # half of them
    _dump(tmp_path / "a.npz", meta, presence=a)
    _dump(tmp_path / "b.npz", meta, presence=b)
    cli(["compare", "--a", str(tmp_path / "a.npz"), "--b", str(tmp_path / "a.npz"), "--out", str(tmp_path / "aa.json")])
    same = json.load(open(tmp_path / "aa.json"))
    assert same["a_minus_b"]["floor_eff1pct_clear_0.1-0.2"] == [0.0, 0.0, 0.0] and same["n_pairs"] == 1
    cli(["compare", "--a", str(tmp_path / "a.npz"), "--b", str(tmp_path / "b.npz"), "--out", str(tmp_path / "ab.json")])
    d = json.load(open(tmp_path / "ab.json"))["a_minus_b"]["floor_eff1pct_clear_0.1-0.2"]
    assert d[0] == pytest.approx(1.0 - (b[is_iso] > 1).mean()) and d[1] <= d[0] <= d[2]
    other = [dict(m, ev=19 - m["ev"]) for m in meta]
    _dump(tmp_path / "c.npz", other, presence=a)
    with pytest.raises(SystemExit, match="differ"):
        cli(["compare", "--a", str(tmp_path / "a.npz"), "--b", str(tmp_path / "c.npz"), "--out", str(tmp_path / "x.json")])
    moved = [dict(m, w0=64) if m["kind"] == "bg" else m for m in meta]        # far windows elsewhere
    _dump(tmp_path / "w.npz", moved, presence=a)
    with pytest.raises(SystemExit, match=r"differ \(w0\)"):
        cli(["compare", "--a", str(tmp_path / "a.npz"), "--b", str(tmp_path / "w.npz"), "--out", str(tmp_path / "x.json")])
    _dump(tmp_path / "f.npz", [dict(m, flag=True) for m in meta], presence=a)
    with pytest.raises(SystemExit, match="only some dumps record flag"):
        cli(["compare", "--a", str(tmp_path / "a.npz"), "--b", str(tmp_path / "f.npz"), "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="no score"):
        cli(["compare", "--a", str(tmp_path / "a.npz"), "--arm-b", "amplitude", "--out", str(tmp_path / "x.json")])


def test_compare_averages_the_seed_files_of_each_arm(tmp_path):
    """Two seeds a side: the pairs' differences are averaged and their spread is the seed
    half-range; one file on a side pairs with every file on the other; a seed scored on other
    windows is refused."""
    from helix.eval.__main__ import main as cli
    rng = np.random.default_rng(2)
    meta = _clear_meta()
    s_bg = rng.random(len(meta))
    is_iso = np.array([m["kind"] == "iso" for m in meta])
    half = np.arange(len(meta)) % 2 == 0
    found = np.where(is_iso, 2.0, s_bg)                  # every clear window found
    some = np.where(is_iso, np.where(half, 2.0, 0.0), s_bg)
    _dump(tmp_path / "a0.npz", meta, presence=found)
    _dump(tmp_path / "a1.npz", meta, presence=found)
    _dump(tmp_path / "b0.npz", meta, presence=found)
    _dump(tmp_path / "b1.npz", meta, presence=some)
    cli(["compare", "--a", str(tmp_path / "a0.npz"), str(tmp_path / "a1.npz"),
         "--b", str(tmp_path / "b0.npz"), str(tmp_path / "b1.npz"), "--out", str(tmp_path / "s.json")])
    r = json.load(open(tmp_path / "s.json"))
    one = 1.0 - (some[is_iso] > 1).mean()                # seed 1's difference; seed 0's is 0
    k = "floor_eff1pct_clear_0.1-0.2"
    assert r["n_pairs"] == 2 and r["a_minus_b"][k][0] == pytest.approx(one / 2, abs=1e-3)
    assert r["seed_half_range"][k] == pytest.approx(one / 2, abs=1e-3)
    cli(["compare", "--a", str(tmp_path / "a0.npz"), "--b", str(tmp_path / "b0.npz"), str(tmp_path / "b1.npz"),
         "--out", str(tmp_path / "s1.json")])
    assert json.load(open(tmp_path / "s1.json"))["a_minus_b"][k][0] == pytest.approx(one / 2, abs=1e-3)
    _dump(tmp_path / "b2.npz", [dict(m, ev=(m["ev"] + 1) % 20) for m in meta], presence=found)
    with pytest.raises(SystemExit, match="differ"):
        cli(["compare", "--a", str(tmp_path / "a0.npz"), str(tmp_path / "a1.npz"),
             "--b", str(tmp_path / "b0.npz"), str(tmp_path / "b2.npz"), "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="pair seed by seed"):
        cli(["compare", "--a", str(tmp_path / "a0.npz"), str(tmp_path / "a1.npz"), "--b", str(tmp_path / "b0.npz"),
             str(tmp_path / "b1.npz"), str(tmp_path / "b0.npz"), "--out", str(tmp_path / "x.json")])


def test_compare_shows_a_difference_carried_by_some_views(tmp_path):
    """B finds every clear window on U and none on V or Y: the pooled difference is positive, its
    views block shows U at 0 and V, Y above it, not of one sign."""
    from helix.eval.__main__ import main as cli
    rng = np.random.default_rng(1)
    meta = _clear_meta(n_ev=30, views=(0, 1, 2), n_bg=20, n_iso=4)
    s_bg = rng.random(len(meta))
    iso = np.array([m["kind"] == "iso" for m in meta])
    g = np.array([m["g"] for m in meta])
    _dump(tmp_path / "a.npz", meta, presence=np.where(iso, 2.0, s_bg))
    _dump(tmp_path / "b.npz", meta, presence=np.where(iso & (g == 0), 2.0, np.where(iso, 0.0, s_bg)))
    cli(["compare", "--a", str(tmp_path / "a.npz"), "--b", str(tmp_path / "b.npz"), "--out", str(tmp_path / "ab.json")])
    v = json.load(open(tmp_path / "ab.json"))["views"]["floor_eff1pct_clear_0.1-0.2"]
    assert v["U"] == 0 and v["V"] > 0.9 and v["Y"] > 0.9 and not v["same_sign"]
    assert v["mean"] == pytest.approx((v["U"] + v["V"] + v["Y"]) / 3, abs=1e-3)


def test_the_cli_writes_the_window_scores_beside_the_json(tmp_path, monkeypatch):
    """A metric's ``_windows`` arrays leave the JSON for ``<out>.windows.npz``."""
    import helix.eval as E
    from helix.eval.__main__ import main
    fake = {"metrics": {"denoise": {"x": 1, "_windows": {"meta/ev": np.arange(3), "score/presence": np.ones(3)}}}}
    monkeypatch.setattr(E, "evaluate", lambda *a, **k: fake)
    out = tmp_path / "s.json"
    assert main(["run", "--out", str(out)]) == 0
    assert "_windows" not in json.load(open(out))["metrics"]["denoise"]
    z = np.load(tmp_path / "s.windows.npz")
    assert set(z.files) == {"meta/ev", "score/presence"}


def test_cell_charge_reads_the_truth_and_zeros_the_rest():
    cells = dict(key=np.array([5, 2, 9], np.int64), q=np.array([50.0, 20.0, 90.0]))
    assert W.cell_charge(cells, [2, 3, 9, 5]).tolist() == [20.0, 0.0, 90.0, 50.0]
    assert W.cell_charge(dict(key=np.zeros(0, np.int64), q=np.zeros(0)), [1]).tolist() == [0.0]
