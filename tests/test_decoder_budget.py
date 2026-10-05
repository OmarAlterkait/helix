"""Decoder-budget options: ``d_dec``, per-band ``dec_frac``, ``n_masks``.

Each changes what a training step spends, not what the model is evaluated on, so
each test pins one of: the default builds exactly the original model; the option
is training-only; the loss stays an estimate of the full objective; masks drawn
for one step are complementary.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from helix.model.head import cat_head_sparse                        # noqa: E402
from test_serial import SMALL, make_batch                           # noqa: E402
from test_varlen import EXACT, _exact_mask, _model                  # noqa: E402


def test_default_and_equal_d_dec_build_the_original_tree():
    base = set(_model().state_dict())
    assert "dec_embed.weight" not in base
    assert set(_model(d_dec=SMALL["d"]).state_dict()) == base


def test_d_dec_shapes_heads_and_refusals():
    m = _model(d_dec=32)
    assert m.dec_embed.weight.shape == (32, SMALL["d"])
    assert m.dec[0].h == 2 and m.dec[0].hd == SMALL["d"] // SMALL["heads"]
    assert m.val_head.in_features == 32 and m.dec_norm.normalized_shape == (32,)
    with pytest.raises(ValueError):
        _model(d_dec=24)                                   # not a multiple of the head dim
    with pytest.raises(ValueError):
        build_fm({**SMALL, "d_dec": 32, "serial": False, "dec_mode": "self"})


@pytest.mark.parametrize("varlen", [False, True])
def test_d_dec_trains_end_to_end(varlen):
    m = _model(d_dec=32, varlen=varlen).train()
    B = make_batch(**EXACT)
    out = m(B)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert m.dec_embed.weight.grad is not None and m.enc[0].qkv.weight.grad.abs().sum() > 0
    feat, rows = m.forward_feat(B, _exact_mask(EXACT["n_cells"]), masked_only=True)
    assert feat.shape == (rows.numel(), 32)


def test_d_dec_varlen_equals_padded_when_nothing_is_padded():
    a, b = _model(d_dec=32), _model(d_dec=32, varlen=True)
    b.load_state_dict(a.state_dict())
    B, m = make_batch(**EXACT), _exact_mask(EXACT["n_cells"])
    with torch.no_grad():
        torch.testing.assert_close(b.forward_feat(B, m), a.forward_feat(B, m), rtol=1e-5, atol=1e-5)


def test_d_dec_has_its_own_mup_multiplier():
    m = _model(d_dec=32, mup=True, d_base=16)              # m = 4 encoder, 2 decoder
    assert (m.m, m.m_dec, m.readout_mult) == (4.0, 2.0, 0.5)
    cats = m._mup_categories()
    assert cats["dec.0.q.weight"] == cats["dec_embed.weight"] == "hidden_dec"
    assert cats["enc.0.qkv.weight"] == "hidden"
    lrs = sorted(g["lr"] for g in m.param_groups(1.0, weight_decay=0.1))
    assert lrs == [0.25, 0.5, 1.0, 1.0]
    seen = [p for g in m.param_groups(1.0, weight_decay=0.1) for p in g["params"]]
    assert len(seen) == len(list(m.parameters())) == len(set(map(id, seen)))
    eq = _model(mup=True, d_base=16)                       # d_dec = d: no fourth group
    assert len(eq.param_groups(1.0, weight_decay=0.1)) == 3


def test_row_weight_two_equals_the_row_twice():
    m = _model()
    B = make_batch(**EXACT)
    feat = torch.randn(3, SMALL["d"])
    rows = torch.tensor([5, 11, 40])
    w = torch.tensor([2.0, 1.0, 1.0])
    a = cat_head_sparse(m, feat, B, rows, w=w)
    b = cat_head_sparse(m, feat[[0, 0, 1, 2]], B, rows[[0, 0, 1, 2]])
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)
    for x, y in zip(cat_head_sparse(m, feat, B, rows, w=torch.ones(3)), cat_head_sparse(m, feat, B, rows)):
        torch.testing.assert_close(x, y)


def test_per_band_dec_frac_keeps_full_bands_and_weights_the_rest():
    m = _model(varlen=True).train()
    m.dec_frac = (1.0, 1.0, 0.5, 0.5)
    B = make_batch(n_cells=4000)
    mask = torch.arange(4000) % 4 != 0
    torch.manual_seed(0)
    _, rows = m.forward_feat(B, mask, masked_only=True)
    band = B["band_id"]
    full = mask & (band < 2)
    assert bool(mask[rows].all())
    assert set(rows[band[rows] < 2].tolist()) == set(full.nonzero().squeeze(1).tolist())
    part = mask & (band >= 2)
    frac = (band[rows] >= 2).sum().item() / part.sum().item()
    assert 0.45 < frac < 0.55
    w = m.dec_weights(B, rows)
    assert torch.equal(w, torch.where(band[rows] < 2, 1.0, 2.0))
    m.eval()
    assert m.dec_weights(B, rows) is None
    with torch.no_grad():
        _, rows_eval = m.forward_feat(B, mask, masked_only=True)
    assert rows_eval.numel() == int(mask.sum())


def test_complementary_masks_are_disjoint_and_sized():
    m = _model().train()
    m.n_masks, m.mask_ratio = 2, 0.75
    B = make_batch(n_cells=8000)
    a, b = m.draw_masks(B, 2)
    assert not bool((~a & ~b).any())                       # no token visible in both
    for x in (a, b):
        assert 0.23 < (~x).float().mean().item() < 0.27
    with pytest.raises(ValueError):
        m.draw_masks(B, 5)                                 # 5 x 25% visible > all tokens


def test_n_masks_averages_losses_in_training_only():
    m = _model(varlen=True).train()
    m.n_masks = 2
    B = make_batch(**EXACT)
    out = m(B)
    assert torch.isfinite(out["loss"]) and set(out) == {"loss", "bce", "val", "masked_frac"}
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(m(B)["loss"])                # eval: one mask, unchanged path
