"""helix.eval: reading a run, the evaluated events and their windows, and every metric."""
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
# set it read, and refuses what it cannot read; the probes are seeded and learn what a feature
# carries (an informative "trained" arm beats its reference: the coefficient amplitude, the
# geometry columns, the token layout); a perfect denoiser scores perfectly; recon is the
# evaluator's metric.

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
    """A run directory: resolved_config.json, provenance.json, the EMA sidecar (WeightEMA._save's
    layout), pimm's split checkpoint model/last (weights.pth + trainer.dcp) and an iter snapshot."""
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
def test_load_run_reads_the_ema_and_the_last_checkpoint(tmp_path):
    raw, ema = _write_run(tmp_path)
    r = load_run(tmp_path, device="cpu")
    assert _same(r.model, ema) and not r.model.training
    assert not any(p.requires_grad for p in r.model.parameters())
    assert torch.isfinite(r.model.bin_edges).all()
    assert r.tokenizer == patch_config(TOK)
    p = r.provenance
    assert (p["weights"], p["step"], p["run"], p["bins"]) == ("ema", 5, str(tmp_path), "/archive/bins.pt")
    assert p["corpus"]["identity"] == {"basis_digest": "abc"} and p["code"]["helix"]["commit"] == "c0ffee"
    r = load_run(tmp_path, "last", device="cpu")
    assert _same(r.model, raw) and (r.provenance["weights"], r.provenance["step"]) == ("last", 7)


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
        load_run(tmp_path, device="cpu")                             # no silent fallback to raw weights
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


def _ctx(run, n, n_boot=20, denoiser=None):
    from helix.eval import Context
    from helix.eval.probe import split_events
    return Context(run, run.model, split_events(n), 0, n_boot, denoiser)


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


def test_recon_is_the_evaluators_metric(synth):
    from helix.eval import recon as R
    d, ds = synth
    run = load_run(d, device="cpu")
    m = R.Recon(_ctx(run, len(ds)))
    evs = _events(run, ds)[:3]
    for e in evs:
        m.add(e)
    got = m.result()
    totals, counts, gf = {}, {}, dict.fromkeys(R.GF_KEYS, 0.0)
    for i, e in enumerate(evs):                                       # the evaluator's loop, by hand
        mask = R.mask_for(run.model, e.batch, i)
        with torch.no_grad():
            R.pool(R.heads_loss(run.model, e.batch, mask, gf), e.batch, mask, totals, counts)
    want = R.average(totals, counts, gf)
    assert got["n_events"] == 3 and {"loss", "bce", "val", "var_expl", "charge_closure", "charge_resid"} <= set(got)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-6)
    # and the same under the region mask, beside it
    totals, counts, gf = {}, {}, dict.fromkeys(R.GF_KEYS, 0.0)
    for e in evs:
        mask = R.region_mask(e.batch, run.tokenizer.pw)
        with torch.no_grad():
            R.pool(R.heads_loss(run.model, e.batch, mask, gf), e.batch, mask, totals, counts)
    want = R.average(totals, counts, gf)
    assert {"loss", "var_expl", "charge_closure", "mask_frac"} <= set(got["region"])
    for k, v in want.items():
        assert got["region"][k] == pytest.approx(v, rel=1e-6)
    assert 0 < got["region"]["mask_frac"] < 1 and got["region"]["var_expl"] != got["var_expl"]


def test_the_region_mask_hides_whole_regions_as_the_predecessor_did():
    """Tokens of one (plane, 32-wire, 128-tick) region are masked together, 75% of regions are,
    and the regions are the predecessor's (scripts/eval_resolution.py's crc32 of the key)."""
    import zlib
    from helix.eval.recon import region_mask
    rng = np.random.default_rng(0)
    n, pw = 20000, 16
    B = dict(cell_wb=torch.from_numpy(rng.integers(0, 120, n)),
             t_phys=torch.from_numpy(rng.uniform(-20, 6000, n).astype(np.float32)),
             plane_id=torch.from_numpy(rng.integers(0, 6, n)))
    m = region_mask(B, pw).numpy()
    wc = B["cell_wb"].float().numpy() * pw + pw / 2
    rk = (B["plane_id"].numpy().astype(np.int64) << 40) | ((wc // 32).astype(np.int64) << 20) \
        | (np.floor(B["t_phys"].numpy() / 128).astype(np.int64) + 4096)
    ref = np.array([zlib.crc32(int(x).to_bytes(8, "little", signed=True)) % 1000 for x in rk]) < 750
    np.testing.assert_array_equal(m, ref)
    u, inv = np.unique(rk, return_inverse=True)
    per = np.bincount(inv, m, len(u)) / np.bincount(inv, minlength=len(u))
    assert set(np.unique(per)) <= {0.0, 1.0} and 0.7 < per.mean() < 0.8


def test_floor_learns_what_the_features_carry(synth):
    from helix.eval.probe import Floor
    d, ds = synth
    run = load_run(d, device="cpu")
    m = Floor(_ctx(run, len(ds)))
    for e in _events(run, ds):
        e._feats = _token_feature(e, np.log1p(e.cells["q"] / C.Q0))
        m.add(e)
    out = m.result()
    assert out["n_train_events"] == 6 and np.isfinite(out["val_mse"])
    for arm in ("trained", "amplitude"):
        r = out[arm]
        assert len(r["floor_eff1pct_0.1-0.2"]) == 3 and min(r["n_windows"].values()) > 0
    # the coefficient values are noise here, so the amplitude finds nothing the features do not
    eff = lambda a: np.nanmean([v[0] for k, v in out[a].items() if k.startswith("floor_eff")])   # noqa: E731
    assert eff("trained") > eff("amplitude") + 0.2, (out["trained"], out["amplitude"])
    k = "floor_eff1pct_0.5-1e+09"
    assert out["d_over_amplitude"][k] == pytest.approx(out["trained"][k][0] - out["amplitude"][k][0], abs=1e-4)


def test_uprobe_learns_u_and_geometry_alone_does_not(synth):
    from helix.eval.probe import UProbe, u_of
    d, ds = synth
    run = load_run(d, device="cpu")
    m = UProbe(_ctx(run, len(ds)))
    for e in _events(run, ds):
        g = C.unkey(e.cells["key"])[0]
        e._feats = _token_feature(e, u_of(g, e.cells["dom_y"], e.cells["dom_z"]))
        m.add(e)
    out = m.result()
    assert out["trained"]["r2"][0] > 0.5 > out["geo"]["r2"][0]
    assert all(out["trained"][f"r2_{v}"][0] > 0.3 for v in "UVY")
    d, lo, hi = out["d_over_geo"]
    assert d == pytest.approx(out["trained"]["r2"][0] - out["geo"]["r2"][0], abs=1e-3) and lo <= d <= hi


def test_labels_learn_the_class_the_features_carry(synth):
    from helix.eval.probe import Labels
    d, ds = synth
    run = load_run(d, device="cpu")
    m = Labels(_ctx(run, len(ds)))
    for e in _events(run, ds):
        e._feats = _token_feature(e, C.cell_class(e.cells, "trk_motif").astype(np.float64))
        m.add(e)
    out = m.result()
    mo = out["motif"]
    assert set(mo["prior"]) == set(C.MOTIF_NAMES) and sum(mo["prior"].values()) == pytest.approx(1, abs=1e-3)
    assert mo["trained"]["macro_recall"][0] > 0.6 > mo["geo"]["macro_recall"][0]
    assert mo["d_over_geo"][0] == pytest.approx(mo["trained"]["macro_recall"][0] - mo["geo"]["macro_recall"][0], abs=1e-3)
    assert set(out["pid"]["trained"]["recall"]) <= set(C.PID_NAMES)


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


def test_a_perfect_denoiser_scores_perfectly(synth):
    from helix.eval.denoise import Denoise
    d, ds = synth
    run = load_run(d, device="cpu")

    def oracle(ev, keys):
        q = dict(zip(ev.cells["key"].tolist(), ev.cells["q"].tolist()))
        idx, _ = C.cell_inputs(keys, ev.cell_key, ev.band_lengths, run.tokenizer)
        ok = (idx >= 0).any(1)
        y = np.log1p(np.array([q.get(k, 0.0) for k in keys.tolist()]) / C.Q0).astype(np.float32)
        y = np.where(ok, y, 0)
        return y, (y > 0).astype(np.float32)

    m = Denoise(_ctx(run, len(ds), denoiser=oracle))
    for e in _events(run, ds):
        m.add(e)
    out = m.result()
    for r in (out["charge"], out["presence"]):
        effs = [v[0] for k, v in r.items() if k.startswith("floor_eff1pct_") and v[0] == v[0]]
        assert effs and min(effs) == 1.0


def _oracle(scale=1.0, q_present=25.0):
    """A denoiser that knows the truth: ``scale`` x each covered cell's charge, presence 1 above
    ``q_present`` e-."""
    def predict(ev, keys):
        q = dict(zip(ev.cells["key"].tolist(), ev.cells["q"].tolist()))
        idx, _ = C.cell_inputs(keys, ev.cell_key, ev.band_lengths, patch_config(TOK))
        t = np.array([q.get(k, 0.0) for k in keys.tolist()]) * ((idx >= 0).any(1))
        return np.log1p(scale * t / C.Q0).astype(np.float32), (t > q_present).astype(np.float32)
    return predict


def test_the_denoise_charge_numbers_are_exact_for_the_truth_and_see_a_scale(synth):
    from helix.eval.denoise import Denoise, PRED_NAMES, TRUTH_NAMES
    d, ds = synth
    run = load_run(d, device="cpu")
    outs = {}
    for scale in (1.0, 1.1):
        m = Denoise(_ctx(run, len(ds), denoiser=_oracle(scale)))
        for e in _events(run, ds):
            m.add(e)
        outs[scale] = m.result()
    for scale, out in outs.items():
        cl = out["cells"]
        assert cl["n_cells"] > 0 and cl["presence_q"] == 25.0
        for v in ("all", "U", "V", "Y"):
            bv = cl["by_view"][v]
            res = [r for r in bv["resolution"].values() if r["n"]]
            assert len(res) >= 2 and sum(r["n"] for r in bv["resolution"].values()) > 0
            for r in res:                                       # log(pred / true) = log(scale), no spread
                assert r["median"] == pytest.approx(np.log(scale), abs=1e-4) and r["hw68"] == pytest.approx(0, abs=1e-4)
                assert r["frac_zero"] == 0
            lin = [r for r in bv["linearity"].values() if r["n"]]
            assert lin and all(r["median_true_over_pred"] == pytest.approx(1 / scale, rel=1e-4) for r in lin)
            assert bv["closure"]["median"] == pytest.approx(scale, rel=1e-4) and bv["closure"]["n_events"] == 3
            cal = cl["calibration"]["presence"][v]                   # presence is right either way
            assert cal["n"] > cl["by_view"][v]["resolution"][TRUTH_NAMES[0]]["n"] and 0 < cal["frac_true"] < 1
            assert cal["ece"] == 0 and cal["brier"] == 0 and cal["log_loss"] < 1e-6
        assert set(cl["by_view"]["all"]["linearity"]) == set(PRED_NAMES)
        assert "floor_eff1pct_near_0.1-0.2" in out["charge"] and "near_fpr_at_far" in out["presence"]


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


def test_fit_probe_is_seeded_and_selects_on_the_metric_it_reports():
    from helix.eval.probe import fit_probe
    from helix.eval.events import group_r2

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


def test_a_classifier_probe_balances_its_classes():
    """A class at 2% of the rows that the inputs separate is still recalled: the loss is
    weighted by inverse class frequency and selection is on the mean per-class recall."""
    from helix.eval.probe import fit_probe
    rng = np.random.default_rng(0)
    n = 6000
    e = np.repeat(np.arange(12), n // 12)
    y = np.where(rng.random(n) < 0.02, 2, rng.integers(0, 2, n))
    X = (rng.standard_normal((n, 16)) + 1.5 * np.eye(16)[y]).astype(np.float32)
    p = fit_probe(torch.from_numpy(X), np.zeros((n, 0), np.float32), y, e, n_classes=3, seed=0, device="cpu")
    got = p.predict(torch.from_numpy(X), np.zeros((n, 0), np.float32))
    assert got.dtype == np.int64 and (got[y == 2] == 2).mean() > 0.6 and 0.6 < p.val_score <= 1


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


def test_the_probes_fit_on_a_budget_and_keep_no_features_of_a_scored_event(synth, monkeypatch):
    import helix.eval.probe as P
    from helix.eval.probe import Floor
    from helix.eval.probe import Labels
    from helix.eval.probe import UProbe
    rng = lambda: np.random.default_rng(3)                                             # noqa: E731
    assert np.array_equal(P.fit_rows(5, 4, rng()), np.arange(5))
    monkeypatch.setattr(P, "FIT_ROWS", 3000)         # 500 rows per training event: binds
    k = P.fit_rows(1000, 7, rng())
    assert len(k) == 428 and np.array_equal(k, np.unique(k)) and np.array_equal(k, P.fit_rows(1000, 7, rng()))
    d, ds = synth
    run = load_run(d, device="cpu")
    ctx = _ctx(run, len(ds), n_boot=5)
    ms = [Floor(ctx), UProbe(ctx), Labels(ctx)]
    for e in _events(run, ds):
        for m in ms:
            m.add(e)
            assert (m.fit is None) == (e.index >= ctx.n_train)      # fitted at the first scored event
    assert P.gather(torch.zeros((5, 3)), np.zeros((0, 4), np.int64)).shape == (0, 12)   # an event without rows
    held = list(ms[0].scored.s["trained"]) + list(ms[1].p["trained"]) + list(ms[2].p[("motif", "trained")])
    assert held and all(isinstance(x, np.ndarray) and x.ndim == 1 for x in held)    # predictions, no features
    for m in ms:
        out = m.result()
        assert 0 < out["n_fit_rows"] <= 3000


def test_the_cli_writes_one_json(synth, tmp_path, monkeypatch):
    import helix.eval as E
    import helix.eval.events as EV
    from helix.eval.__main__ import main
    d, ds = synth
    monkeypatch.setattr(EV, "dataset", lambda run: ds)
    out = tmp_path / "s.json"
    assert main([str(d), "--metrics", "recon", "--out", str(out), "--max-events", "2"]) == 0
    res = json.load(open(out))
    assert res["n_events"] == 2 and set(res["metrics"]) == {"recon"} and res["run"]["weights"] == "ema"
    with pytest.raises(ValueError, match="unknown metric"):
        E.evaluate(d, ["nope"])


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
    assert set(kinds) == {"iso", "bg", "near"}
    assert kinds.count("bg") <= W.N_BG_PLANE * 1 and kinds.count("near") <= W.N_BG_PLANE    # one plane here
    assert len(tr["wkey"]) == len(tr["meta"]) * N_CELL
    q = _charge(rec, tr["wkey"])
    near_charge = _haze(rec, tr["wkey"], 1.0) > 0                 # charge within 8 cells
    for i, m in enumerate(tr["meta"]):
        cells = slice(i * N_CELL, (i + 1) * N_CELL)
        assert np.array_equal(tr["wkey"][cells], W.window_cells(m["g"], m["w0"], m["t0"]))
        if m["kind"] == "bg":
            assert q[cells].sum() == 0 and np.isnan(m["E"]) and not near_charge[cells].any()
        elif m["kind"] == "near":                                  # empty, a kept coefficient, charge beside
            assert q[cells].sum() == 0 and np.isnan(m["E"]) and near_charge[cells].any()
            assert np.isin(tr["wkey"][cells], ck).any()
        else:
            assert m["E"] > 0 and q[cells].sum() > 0
    # the near windows are drawn last: iso and bg are what they were without them
    from helix.eval import events as EV
    import unittest.mock as um
    with um.patch("helix.data.denoise.near_windows", lambda *a, **k: []):
        old = EV.event_truth(rec, ck, np.random.default_rng(1))
    assert old["meta"] == [m for m in tr["meta"] if m["kind"] != "near"]


def test_every_plane_gets_its_own_background():
    rec, ck = _event(np.random.default_rng(0))
    g, fw, ft = C.unkey(ck)
    ck = np.concatenate([ck, C.fkey(np.full(len(ck), 2), fw, ft)])      # a second plane: noise only
    tr = W.event_truth(rec, ck, np.random.default_rng(1))
    bg = [m["g"] for m in tr["meta"] if m["kind"] == "bg"]
    assert bg.count(0) == W.N_BG_PLANE and bg.count(2) == W.N_BG_PLANE


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
    assert out["near_fpr_at_far"][0] == 0.0 and out["n_windows"]["near_U"] > 0
    aucs = [v[0] for k, v in out.items() if k.startswith("auc_") and v[0] == v[0]]
    assert aucs and all(v == 1.0 for v in aucs)
    assert np.isnan(out["floor_eff1pct_0.1-0.2_V"][0])                  # no V window: not a 0


def test_a_haze_around_tracks_is_free_on_the_far_background_and_priced_on_the_near():
    """A prediction that puts charge (above any particle's) on every cell within 8 cells of truth
    detects every isolated particle against the far background, which keeps those surroundings
    empty by construction -- and nothing against the near background, on which the far threshold
    fires every time."""
    rec, ck = _event(np.random.default_rng(0))
    tr = W.event_truth(rec, ck, np.random.default_rng(1))
    meta = [dict(m, ev=0) for m in tr["meta"]]
    q = _charge(rec, tr["wkey"])
    hazy = np.maximum(q, _haze(rec, tr["wkey"], 1.5 * q.max()))
    good, bad = (W.score_windows(s, meta, n_boot=5) for s in (q, hazy))
    E = [nm for nm in W.ENERGY_NAMES if good[f"floor_eff1pct_{nm}"][0] == good[f"floor_eff1pct_{nm}"][0]]
    assert E
    for nm in E:
        assert bad[f"floor_eff1pct_{nm}"][0] == 1.0 == good[f"floor_eff1pct_{nm}"][0]           # far: blind to it
        assert bad[f"floor_eff1pct_{nm}_U"][0] == 1.0
        assert bad[f"floor_eff1pct_near_{nm}"][0] == 0.0 and good[f"floor_eff1pct_near_{nm}"][0] == 1.0
        assert bad[f"floor_eff1pct_near_{nm}_U"][0] == 0.0
    assert bad["near_fpr_at_far"][0] == 1.0 == bad["near_fpr_at_far_U"][0] and good["near_fpr_at_far"][0] == 0.0


def test_the_auc_is_the_weighted_pair_ordering_within_a_view():
    rng = np.random.default_rng(0)
    meta, score = [], []
    for g, shift in ((0, 0.5), (2, 1.0)):
        for _ in range(80):
            meta.append(dict(kind="bg", g=g, E=np.nan, ev=len(meta) % 10))
            score.append(rng.random())
        for _ in range(40):
            meta.append(dict(kind="iso", g=g, E=0.15, ev=len(meta) % 10))
            score.append(shift * rng.random() + 0.3 * shift)
    score = np.round(score, 2)                                          # ties count one half
    stat, n, _ = W.window_stat(np.repeat(score, N_CELL), meta)
    view = np.array([m["g"] % 3 for m in meta]); kind = np.array([m["kind"] for m in meta])
    ev = np.array([m["ev"] for m in meta])

    def brute(m, views):
        num = den = 0.0
        for v in views:
            i, b = np.nonzero((kind == "iso") & (view == v))[0], np.nonzero((kind == "bg") & (view == v))[0]
            w = np.outer(m[ev[i]], m[ev[b]])
            num += (w * ((score[i][:, None] > score[b][None]) + 0.5 * (score[i][:, None] == score[b][None]))).sum()
            den += w.sum()
        return num / den
    m = rng.integers(0, 3, n)                                           # a resample's weights
    got = stat(m)
    assert got["auc_0.1-0.2_U"] == pytest.approx(brute(m, [0])) and got["auc_0.1-0.2_Y"] == pytest.approx(brute(m, [2]))
    assert got["auc_0.1-0.2"] == pytest.approx(brute(m, [0, 2]))      # pooled over views: pairs within a view


def test_the_ev_efficiency_weighs_every_event_equally():
    """One event with 50 detected windows and four with one missed each: pooled 50/54, per event
    1/5."""
    meta, score = [], []
    for e in range(5):
        meta += [dict(kind="bg", g=0, E=np.nan, ev=e)] * 20
        score += [0.0] * 20
    meta += [dict(kind="iso", g=0, E=0.15, ev=0)] * 50
    score += [1.0] * 50
    for e in range(1, 5):
        meta.append(dict(kind="iso", g=0, E=0.15, ev=e))
        score.append(0.0)
    out = W.score_windows(np.repeat(score, N_CELL), meta, n_boot=5)
    assert out["floor_eff1pct_0.1-0.2"][0] == pytest.approx(50 / 54, abs=1e-4)
    assert out["floor_eff1pct_0.1-0.2_ev"][0] == pytest.approx(0.2, abs=1e-4)


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
            meta.append(dict(kind="iso", g=g, E=0.15, ev=int(rng.integers(0, 30))))
            score.append(scale * (1.5 + rng.random()))
    out = W.score_windows(np.repeat(score, N_CELL), meta, n_boot=5)
    assert out["floor_eff1pct_0.1-0.2"][0] == 1.0
    pooled = np.quantile([s for s, m in zip(score, meta) if m["kind"] == "bg"], 0.99)
    assert np.mean([s > pooled for s, m in zip(score, meta) if m["kind"] == "iso"]) < 0.7   # what one pooled cut gives


def test_group_r2_ignores_offsets_and_charges_scale():
    rng = np.random.default_rng(0)
    grp = np.repeat(np.arange(10), 50)
    y = rng.standard_normal(500) + 10 * grp                 # between-group variance dominates
    r2 = lambda p: (lambda k, sse, sst: 1 - sse.sum() / sst.sum())(*W.group_r2(p, y, grp))   # noqa: E731
    assert r2(y + 7 * rng.standard_normal(10)[grp]) == pytest.approx(1.0)      # a per-group offset is free
    assert r2(np.zeros(500)) == pytest.approx(0.0) and r2(10 * grp) == pytest.approx(0.0)
    assert r2(3 * y) < 0                                 # perfectly correlated, wrong scale
    keys, sse, sst = W.group_r2(y, y, grp)
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
