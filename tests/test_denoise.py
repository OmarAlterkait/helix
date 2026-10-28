"""The denoiser: its labels (helix.data.denoise.DenoiseRows), its model (helix.model.denoise) and
reading its run back (helix.model.checkpoint.load_run, helix.eval.denoise.load_denoiser).

Synthetic events (tests/test_eval.py's: cells truth + coefficients under it) and an FM run laid
out as pimm writes one. Pinned: the transform's query cells are covered by a token, carry their
full truth charge and a presence label at the support threshold, include every charged cell, and
are fixed by a seed; the model builds from an FM run with its architecture, tokenizer and (when
pretrained) weights; ``freeze_encoder`` leaves only the head trainable; the param groups cover
exactly the trainable parameters with the head at its own LR; ``cond_charge`` regresses charged
cells only; a denoiser run reads back with its FM run's tokenizer.
"""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helix.data import cells as C                                   # noqa: E402
from helix.data.denoise import KEYS, DenoiseRows, kept_coefficients  # noqa: E402
from helix.model.denoise import ENCODER_PREFIXES, CellDecoder, build_denoise  # noqa: E402
from helix.model.tokenize import CoeffTokenize, PatchConfig, _name_hash  # noqa: E402
from test_eval import SMALL, TOK, _DS, _sample, _same, _write_run  # noqa: E402

CFG = PatchConfig(**TOK)


@pytest.fixture(scope="module")
def fm_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("fm")
    _, ema = _write_run(d)
    return d, ema


@pytest.fixture(scope="module")
def sample():
    return _sample(np.random.default_rng(3), 0)


def _rows(s, **kw):
    s = CoeffTokenize(part="coeff", cfg=CFG)(dict(s))
    return DenoiseRows(part="coeff", **kw)(s)


def _batch(s):
    """CoeffCollect's batch: the token part's arrays as tensors."""
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in s["coeff"].items() if isinstance(v, np.ndarray)}


# ------------------------------------------------------------------------------------ transform
def test_the_tokens_hold_the_kept_coefficients(sample):
    c = sample["coeff"]
    tok = CoeffTokenize(part="coeff", cfg=CFG)(dict(sample))["coeff"]
    got = np.stack(kept_coefficients(tok, CFG), 1)
    k = c["band"] < CFG.n_bands
    want = np.stack([c["band"][k], c["plane_gid"][k], c["wire"][k], c["tau"][k]], 1).astype(np.int64)
    np.testing.assert_array_equal(got, want)                       # same rows, same order


def test_query_cells_shapes_and_labels(sample):
    s = _rows(sample, seed=0)
    t, rec = s["coeff"], sample["cells"]
    n, nb = len(t["q_y"]), CFG.n_bands
    assert n > 0 and t["q_idx"].shape == (n, nb) and t["q_aux"].shape == (n, 13 * nb)
    assert t["q_nbr"].shape == (n, nb, 9)
    assert (t["q_idx"] >= 0).any(1).all()                         # every query cell is covered
    np.testing.assert_array_equal(t["q_nbr"][:, :, 4], t["q_idx"])  # slot 4 is the covering token
    # the labels are the cells' truth: recover each row's key from the transform's own draw
    rng = np.random.default_rng([0, _name_hash(sample["name"])])
    keys, q = DenoiseRows(seed=0).rows(kept_coefficients(t, CFG), rec, CFG, rng)
    idx, _ = C.cell_inputs(keys, t["cell_key"], t["_meta"]["band_lengths"], CFG)
    ok = (idx >= 0).any(1)
    np.testing.assert_array_equal(idx[ok], t["q_idx"])
    np.testing.assert_allclose(t["q_y"], np.log1p(q[ok] / C.Q0).astype(np.float32))
    truth = dict(zip(rec["key"].tolist(), rec["q"].tolist()))
    np.testing.assert_allclose(q, [truth.get(k, 0.0) for k in keys.tolist()])
    np.testing.assert_array_equal(t["q_present"], (t["q_y"] > 0).astype(np.float32))
    covered = (C.cell_inputs(rec["key"], t["cell_key"], t["_meta"]["band_lengths"], CFG)[0] >= 0).any(1)
    assert set(rec["key"][covered & (rec["q"] > 0)].tolist()) <= set(keys[ok].tolist())   # every charged cell
    assert (t["q_present"] == 0).sum() > (t["q_present"] == 1).sum()  # and more empty ones beside them


def test_a_seed_fixes_the_draw_and_none_draws_afresh(sample):
    a, b = _rows(sample, seed=0)["coeff"], _rows(sample, seed=0)["coeff"]
    for k in KEYS:
        np.testing.assert_array_equal(a[k], b[k])
    c, d = _rows(sample)["coeff"], _rows(sample)["coeff"]
    assert len(c["q_y"]) != len(d["q_y"]) or not np.array_equal(c["q_aux"], d["q_aux"])


def test_support_threshold_decides_charged_not_the_target():
    """Sampling and presence count a cell as charged only above ``support_min_q``; the
    regression target stays its full charge."""
    tkeys = C.fkey(np.zeros(3, np.int64), np.array([10, 20, 30]), np.array([5, 5, 5]))
    cells = dict(key=tkeys, q=np.array([0.5, 30.0, 4000.0]))        # a tail, a faint cell, a bright cell
    coeffs = tuple(np.zeros(1, np.int64) for _ in range(4))
    kw = dict(cov_per_sig=0, win_per_event=0, near_any_per_sig=0, win_near_per_event=0)
    rng = np.random.default_rng
    k0, _ = DenoiseRows(support_min_q=0.0, **kw).rows(coeffs, cells, CFG, rng(0))
    k25, q25 = DenoiseRows(support_min_q=25.0, **kw).rows(coeffs, cells, CFG, rng(0))
    assert set(tkeys.tolist()) <= set(k0.tolist())
    assert tkeys[0] not in k25 and {tkeys[1], tkeys[2]} <= set(k25.tolist())
    np.testing.assert_allclose(q25[np.isin(k25, tkeys)], [30.0, 4000.0])


def test_the_transform_needs_tokens_and_cells(sample):
    with pytest.raises(KeyError, match="CoeffTokenize"):
        DenoiseRows()(dict(sample))
    s = CoeffTokenize(part="coeff", cfg=CFG)(dict(sample))
    s.pop("cells")
    with pytest.raises(KeyError, match="cells"):
        DenoiseRows()(s)


# ---------------------------------------------------------------------------------------- model
def test_the_model_is_the_fm_runs_encoder(fm_run):
    d, ema = fm_run
    m = build_denoise(str(d), compile_blocks=False)
    assert m.tokenizer == CFG and m.fm.d == SMALL["d"]
    enc = {k: v for k, v in m.fm.state_dict().items()}
    assert all(torch.equal(enc[k].float(), v.float()) for k, v in ema.state_dict().items())   # its EMA
    s = build_denoise(str(d), pretrained=False, compile_blocks=False)
    assert not _same(s.fm, ema) and s.tokenizer == CFG
    assert not torch.isfinite(s.fm.bin_edges).any()                 # from scratch: nothing of the run


def test_param_groups_and_freeze_encoder(fm_run):
    d, _ = fm_run
    for freeze in (False, True):
        m = build_denoise(str(d), freeze_encoder=freeze, lr_head=7e-3, compile_blocks=False)
        groups = m.param_groups(1e-3, weight_decay=0.05)
        seen = [p for g in groups for p in g["params"]]
        train = [p for p in m.parameters() if p.requires_grad]
        assert len(seen) == len(train) and {id(p) for p in seen} == {id(p) for p in train}
        assert {id(p) for p in groups[-1]["params"]} == {id(p) for p in m.head.parameters()}
        assert groups[-1]["lr"] == 7e-3 and groups[-1]["weight_decay"] == 0.05
        enc = [n for n, p in m.fm.named_parameters() if p.requires_grad]
        if freeze:
            assert not enc and len(groups) == 1
        else:
            assert enc and all(n.startswith(ENCODER_PREFIXES) for n in enc)
            assert any(g["lr"] == 1e-3 / m.fm.m for g in groups[:-1])          # muP's hidden group


@pytest.mark.parametrize("freeze", [False, True])
def test_forward_backward_trains_what_it_should(fm_run, sample, freeze):
    d, _ = fm_run
    torch.manual_seed(0)
    m = build_denoise(str(d), freeze_encoder=freeze, compile_blocks=False).train()
    out = m(_batch(_rows(sample, seed=0)))
    assert set(out) == {"loss", "mse", "bce"} and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert m.head.kv.weight.grad.abs().sum() > 0
    g = m.fm.enc[0].qkv.weight.grad
    assert (g is None) if freeze else (g.abs().sum() > 0)
    assert all(p.grad is None for n, p in m.fm.named_parameters() if n.startswith(("dec", "occ_head", "val_head")))


def test_cond_charge_regresses_charged_cells_only(fm_run, sample):
    d, _ = fm_run
    B = _batch(_rows(sample, seed=0))
    torch.manual_seed(0)
    m = build_denoise(str(d), cond_charge=True, compile_blocks=False).eval()
    with torch.no_grad():
        y, _ = m.predict(B, B["q_idx"], B["q_aux"], B["q_nbr"])
        out = m(B)
        pos = B["q_present"] > 0
        torch.testing.assert_close(out["mse"], ((y[pos] - B["q_y"][pos]) ** 2).mean())
        m.cond_charge = False
        torch.testing.assert_close(m(B)["mse"], ((y - B["q_y"]) ** 2).mean())


def test_cell_decoder_survives_cells_with_no_tokens_and_is_unbounded():
    """No normalisation between the residual stream and the readout: a LayerNorm there capped the
    charge logit (M5d / M5dc saturated near 0.8M e- per cell)."""
    torch.manual_seed(0)
    head = CellDecoder(SMALL["d"], n_bands=4, n_slot=SMALL["n_slot"], proj=16, heads=4, layers=1, hidden=32)
    n_tok = 50
    B = dict(inp=torch.randn(n_tok, SMALL["n_slot"]), occ=(torch.rand(n_tok, SMALL["n_slot"]) > 0.5).float())
    idx = torch.randint(-1, n_tok, (20, 4)); nbr = torch.randint(-1, n_tok, (20, 4, 9))
    idx[:3] = -1; nbr[:3] = -1                                      # no token anywhere near
    aux, feats = torch.randn(20, 52), torch.randn(n_tok, SMALL["d"])
    with torch.no_grad():
        y1, p1 = head(feats, idx, aux, nbr, B)
        assert torch.isfinite(y1).all() and torch.isfinite(p1).all()
        head.q0[-1].bias += 1000.0                                  # a large residual stream...
        y2, _ = head(feats, idx, aux, nbr, B)
    assert (y2 - y1).abs().max() > 10                               # ...must reach the output


# ------------------------------------------------------------------------------- reading it back
def _write_denoise_run(d, fm_dir):
    """A denoiser run as pimm writes one: its config names the FM run and a tokenizer by
    reference; model/model_best.pth holds its weights."""
    os.makedirs(os.path.join(d, "model"), exist_ok=True)
    cfg = {"model": {"type": "Coeff-Denoise", "fm": str(fm_dir), "pretrained": True, "freeze_encoder": False,
                     "cond_charge": False, "lr_head": 1e-2, "compile_blocks": True},
           "data": {"train": {"type": "CoeffTPCDataset", "data_root": "/nowhere", "split": ["run_x"],
                              "transform": [{"type": "CoeffTokenize", "part": "coeff", "cfg": "model"},
                                            {"type": "DenoiseRows", "part": "coeff"},
                                            {"type": "CoeffCollect", "part": "coeff"}]}}}
    json.dump(cfg, open(os.path.join(d, "resolved_config.json"), "w"))
    torch.manual_seed(5)
    m = build_denoise(str(fm_dir), compile_blocks=False)
    torch.save({"state_dict": {"module." + k: v for k, v in m.state_dict().items()}},
               os.path.join(d, "model", "model_best.pth"))
    return m


def test_a_denoiser_run_reads_back_with_its_fm_runs_tokenizer(fm_run, sample, tmp_path):
    from helix.eval.denoise import load_denoiser
    from helix.eval.events import events
    from helix.model.checkpoint import load_run

    d, _ = fm_run
    m = _write_denoise_run(str(tmp_path), d)
    r = load_run(tmp_path, "best", device="cpu")
    assert _same(r.model, m) and r.tokenizer == CFG and r.provenance["weights"] == "best"
    with pytest.raises(FileNotFoundError, match="model_ema"):
        load_run(tmp_path, device="cpu")                           # a denoiser has no EMA
    predict, cfg = load_denoiser(str(tmp_path), "cpu")
    assert cfg == CFG
    ev = next(events(_DS([sample]), CFG, torch.device("cpu")))
    y, pr, ok = predict(ev, sample["cells"]["key"])
    assert y.shape == pr.shape == ok.shape == (len(sample["cells"]["key"]),) and ok.any()
    assert ((pr >= 0) & (pr <= 1)).all() and (y[~ok] == 0).all()
    with pytest.raises(ValueError, match="not a denoiser"):
        load_denoiser(str(d), "cpu", weights="ema")


def test_the_fm_run_is_required_to_be_an_fm(tmp_path, fm_run):
    d, _ = fm_run
    _write_denoise_run(str(tmp_path), d)
    with pytest.raises(ValueError, match="not an FM run"):
        build_denoise(str(tmp_path))


# --------------------------------------------------------------------------------------- config
def _need_pimm():
    import importlib.util
    if importlib.util.find_spec("pimm") is None:
        pytest.skip("pimm not importable in this env")


def test_config_is_the_recipe_and_resolves_its_tokenizer(fm_run):
    """configs/pimm/denoise.py loads, states the M6d recipe, and its tokenizer reference is filled
    from the model built on the FM run (FMTrainer's resolve_tokenizer)."""
    _need_pimm()
    from pimm.utils.config import Config
    from helix.paths import SiteError
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        cfg = Config.fromfile(os.path.join(root, "configs", "pimm", "denoise.py"))
    except (SiteError, FileNotFoundError) as exc:
        pytest.skip(f"config needs a root this site does not configure: {exc}")
    m = cfg.model
    assert m["type"] == "Coeff-Denoise" and "bins" not in m and m["pretrained"] and not m["freeze_encoder"]
    assert not m["cond_charge"] and m["lr_head"] == 1e-2 and cfg.optimizer["lr"] == 3e-3
    assert cfg.scheduler["type"] == "WarmupCosineLR" and cfg.scheduler["warmup"] == "auto"
    assert cfg.train["type"] == "FMTrainer" and cfg.batch_size == cfg.batch_size_val
    for split in ("train", "val"):
        dd = cfg.data[split]
        assert tuple(dd["modalities"]) == ("coeff", "cells") and dd["holdout"] == cfg.HOLDOUT
        assert [t["type"] for t in dd["transform"]] == ["CoeffTokenize", "DenoiseRows", "CoeffCollect"]
    assert cfg.data["val"]["transform"][1]["seed"] == 0 and "seed" not in cfg.data["train"]["transform"][1]
    hooks = [h["type"] for h in cfg.hooks]
    assert hooks.index("DenoiseEvaluator") < hooks.index("CheckpointSaver")
    assert "WeightEMA" not in hooks and "CoeffFMEvaluator" not in hooks and "FiniteGuard" in hooks
    saver = cfg.hooks[hooks.index("CheckpointSaver")]
    assert saver["evaluator_every_n_steps"] == "auto"              # model_best by validation loss

    from helix.integrations.pimm.trainer import resolve_tokenizer
    from pimm.models.builder import MODELS
    import helix.integrations.pimm  # noqa: F401
    d, _ = fm_run
    model = MODELS.build(dict(m, fm=str(d), compile_blocks=False))
    assert resolve_tokenizer(cfg.data, model.tokenizer)
    for split in ("train", "val", "test"):
        assert PatchConfig(**cfg.data[split]["transform"][0]["cfg"]) == CFG
    assert not resolve_tokenizer(cfg.data, model.tokenizer)        # nothing left to resolve
    with pytest.raises(ValueError, match="tokenizer"):
        resolve_tokenizer({"train": {"transform": [{"type": "CoeffTokenize", "cfg": "model"}]}}, None)


def test_warmup_cosine_schedule():
    _need_pimm()
    from helix.integrations.pimm.trainer import WarmupCosineLR
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    f = WarmupCosineLR(opt, total_steps=110, warmup=10, floor=0.02).lr_lambdas[0]
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0) and f(10) == pytest.approx(1.0)
    assert f(60) == pytest.approx(0.51) and f(110) == pytest.approx(0.02) and f(500) == pytest.approx(0.02)
