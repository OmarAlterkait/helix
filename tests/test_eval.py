"""helix.model.checkpoint.load_run and the helix.eval metrics, on CPU.

The loader reads a run directory laid out the way pimm and helix's hooks write one (pimm's own
split-checkpoint writer when pimm is importable); the metrics run on synthetic events whose
truth is a cells record and whose tokens come from the training tokenizer. Pinned: the loader
returns exactly the saved weights, the tokenizer of the train split and the step of the weight
set it read, and refuses what it cannot read; the probes are seeded and learn what a feature
carries (an informative "trained" arm beats its reference: the coefficient amplitude, the
geometry columns, the token layout); a perfect denoiser scores perfectly; recon is the
evaluator's metric.
"""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helix.data import cells as C                                   # noqa: E402
from helix.model.checkpoint import load_run, run_dir, tokenizer_config  # noqa: E402
from helix.model.tokenize import PatchConfig, patch_config, tau_of_tick  # noqa: E402

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


def test_floor_learns_what_the_features_carry(synth):
    from helix.eval.floor import Floor
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
    from helix.eval.uprobe import UProbe, u_of
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
    from helix.eval.labels import Labels
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
    from helix.eval.uprobe import patch_rows, u_of
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


def test_fit_probe_is_seeded_and_selects_on_the_metric_it_reports():
    from helix.eval.probe import fit_probe
    from helix.eval.windows import group_r2

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
    from helix.eval.floor import Floor
    from helix.eval.labels import Labels
    from helix.eval.uprobe import UProbe
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
