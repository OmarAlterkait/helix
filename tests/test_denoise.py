"""helix.model.denoise: the FM encoder plus a trainable per-cell head.

Pins: the model builds from an FM architecture, with or without pretrained
weights; the cells' covering tokens feed the head (an absent band contributes
zeros); only encoder and head parameters train, so DDP never waits on a gradient
that cannot arrive; the optimizer groups cover exactly the trainable parameters.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from helix.model.denoise import CellHead, build_denoise             # noqa: E402
from test_serial import SMALL, make_batch                           # noqa: E402

N_CELLS = 200


def _model(**kw):
    torch.manual_seed(0)
    return build_denoise(dict(SMALL), **kw)


def _inputs(B, n=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(-1, B["plane_id"].shape[0], (n, SMALL["n_band"]), generator=g)
    aux = torch.randn(n, 13 * SMALL["n_band"], generator=g)
    return idx, aux


def test_forward_backward_and_frozen_decoder():
    m = _model().train()
    B = make_batch(n_cells=N_CELLS)
    idx, aux = _inputs(B)
    y = m(B, idx, aux)
    assert y.shape == (50,) and torch.isfinite(y).all()
    y.square().mean().backward()
    for n, p in m.named_parameters():
        if n.startswith("fm.dec") or n.startswith(("fm.occ_head", "fm.val_head", "fm.mask_tok", "fm.dec_norm")):
            assert not p.requires_grad and p.grad is None, n
    assert m.fm.enc[0].qkv.weight.grad.abs().sum() > 0
    assert m.head.mlp[0].weight.grad.abs().sum() > 0


def test_absent_band_contributes_zeros():
    head = CellHead(SMALL["d"], n_bands=4, proj=8, hidden=16)
    feats = torch.randn(10, SMALL["d"])
    idx = torch.tensor([[0, -1, -1, -1], [0, 3, -1, -1]])
    z = head.proj(head.norm(feats))
    g = z[idx.clamp(min=0)] * (idx >= 0).unsqueeze(-1).float()
    assert torch.equal(g[0, 1:], torch.zeros(3, 8)) and torch.equal(g[1, 1], z[3])


def test_param_groups_cover_exactly_the_trainable_parameters():
    m = _model()
    groups = m.param_groups(1e-3, 2e-3, 0.05)
    seen = [p for g in groups for p in g["params"]]
    train = [p for p in m.parameters() if p.requires_grad]
    assert len(seen) == len(train) and {id(p) for p in seen} == {id(p) for p in train}
    assert groups[-1]["lr"] == 2e-3                                   # the head's group


def test_pretrained_weights_load_into_the_encoder():
    torch.manual_seed(1)
    fm = build_fm(dict(SMALL))
    m = build_denoise(dict(SMALL), fm.state_dict())
    torch.testing.assert_close(m.fm.enc[0].qkv.weight, fm.enc[0].qkv.weight)
    partial = {k: v for k, v in fm.state_dict().items() if not k.startswith("enc.0.")}
    with pytest.raises(ValueError):                                   # encoder weights missing: refuse
        build_denoise(dict(SMALL), partial)


def test_presence_head_returns_charge_and_logit_and_trains():
    m = _model(head_kw=dict(presence=True)).train()
    B = make_batch(n_cells=N_CELLS)
    idx, aux = _inputs(B)
    y, logit = m(B, idx, aux)
    assert y.shape == logit.shape == (50,)
    loss = y.square().mean() + torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.ones(50))
    loss.backward()
    assert m.head.mlp[-1].weight.grad.shape == (2, m.head.mlp[-1].in_features)
    assert set(_model().state_dict()) == set(m.state_dict())             # same tree, wider last layer


def test_cell_decoder_trains_and_survives_cells_with_no_tokens():
    from helix.model.denoise import CellDecoder
    torch.manual_seed(0)
    m = build_denoise(dict(SMALL), head_kw=dict(kind="decoder", presence=True, proj=32, layers=2, hidden=64)).train()
    assert isinstance(m.head, CellDecoder)
    B = make_batch(n_cells=N_CELLS)
    idx, aux = _inputs(B)
    g = torch.Generator().manual_seed(1)
    nbr = torch.randint(-1, B["plane_id"].shape[0], (50, SMALL["n_band"], 9), generator=g)
    nbr[:5] = -1; idx[:5] = -1                                     # cells with no token anywhere near
    y, logit = m(B, idx, aux, nbr)
    assert y.shape == logit.shape == (50,) and torch.isfinite(y).all() and torch.isfinite(logit).all()
    (y.square().mean() + logit.square().mean()).backward()
    assert m.head.kv.weight.grad.abs().sum() > 0 and m.fm.enc[0].qkv.weight.grad.abs().sum() > 0
    groups = m.param_groups(1e-3, 2e-3, 0.05)                      # the head's params are in the optimizer
    assert {id(p) for p in m.head.parameters()} <= {id(p) for g_ in groups for p in g_["params"]}


def test_cell_decoder_output_is_not_bounded():
    """No normalisation between the residual stream and the readout: a LayerNorm there
    capped the charge logit (M5d / M5dc saturated near 0.8M e- per cell)."""
    from helix.model.denoise import CellDecoder
    torch.manual_seed(0)
    head = CellDecoder(SMALL["d"], n_bands=4, n_slot=SMALL["n_slot"], proj=16, heads=4, layers=1, hidden=32)
    B = make_batch(n_cells=N_CELLS)
    idx, aux = _inputs(B, n=20)
    nbr = torch.randint(-1, N_CELLS, (20, 4, 9))
    feats = torch.randn(N_CELLS, SMALL["d"])
    with torch.no_grad():
        y1 = head(feats, idx, aux, nbr=nbr, B=B)
        head.q0[-1].bias += 1000.0                              # a large residual stream...
        y2 = head(feats, idx, aux, nbr=nbr, B=B)
    assert (y2 - y1).abs().max() > 10                           # ...must reach the output
