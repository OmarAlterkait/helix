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


def test_charge_loss_weighting():
    """charge_loss: POW 0 is the plain mean over present rows (the old expression); POW k weights each present row
    by (1 + q / Q0) ** k = exp(k y), normalised; absent rows never count; the weighting is not differentiated."""
    from helix.model.denoise import charge_loss
    g = torch.Generator().manual_seed(0)
    y = torch.rand(64, generator=g) * 6
    p = (y + 0.1 * torch.randn(64, generator=g)).requires_grad_(True)
    pos = y > 1.0
    old = ((p - y) ** 2 * pos).sum() / pos.sum().clamp(min=1)
    assert torch.allclose(charge_loss(p, y, pos, 0.0), old)
    for k in (1.0, 2.0):
        w = torch.exp(k * y) * pos
        ref = ((p - y) ** 2 * w).sum() / w.sum()
        assert torch.allclose(charge_loss(p, y, pos, k), ref, rtol=1e-5)
    p2 = p.detach().clone(); p2[~pos] += 100.0                     # absent rows: no effect
    assert torch.allclose(charge_loss(p2, y, pos, 2.0), charge_loss(p.detach(), y, pos, 2.0))
    p.grad = None; charge_loss(p, y, pos, 2.0).backward()
    gref = 2 * (p.detach() - y) * torch.exp(2 * y) * pos / (torch.exp(2 * y) * pos).sum()
    assert torch.allclose(p.grad, gref, rtol=1e-4, atol=1e-8)
    assert charge_loss(p.detach(), y, torch.zeros_like(pos), 2.0) == 0


def _decoder_inputs(B, n=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx, aux = _inputs(B, n, seed)
    nbr = torch.randint(-1, B["plane_id"].shape[0], (n, SMALL["n_band"], 9), generator=g)
    return idx, aux, nbr


@pytest.mark.parametrize("dist", ["gauss", "hlgauss"])
def test_dist_head_starts_where_the_point_head_is(dist):
    """A point checkpoint loads into a distributional head (only dist_out missing) and init_dist_from_point makes it
    start at the point head: gauss -- same mean, log sigma 0; hlgauss -- the bins' mean = the point charge (a
    Gaussian of width init_sigma around it, inside the binned range). Presence is untouched."""
    kw = dict(kind="decoder", presence=True, proj=32, layers=1, hidden=64)
    torch.manual_seed(0)
    pt = build_denoise(dict(SMALL), head_kw=kw).eval()
    with torch.no_grad():                                          # put the point charge well inside [0, y_hi]
        pt.head.out.bias[0] += 3.0
    dm = build_denoise(dict(SMALL), head_kw=dict(kw, dist=dist, n_bins=512)).eval()
    missing, unexpected = dm.load_state_dict(pt.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("head.dist_out.") for k in missing)
    dm.head.init_dist_from_point()
    B = make_batch(n_cells=N_CELLS)
    idx, aux, nbr = _decoder_inputs(B)
    with torch.no_grad():
        y0, l0 = pt(B, idx, aux, nbr)
        y1, l1, extra = dm(B, idx, aux, nbr)
    assert torch.allclose(l0, l1)
    if dist == "gauss":
        assert torch.allclose(y0, y1) and torch.all(extra == 0)
    else:
        assert extra.shape == (50, 512)
        assert torch.allclose(y0, y1, atol=2e-3)                   # mean of a Gaussian of width 0.05 on 0.0146 bins


def test_gauss_nll_loss():
    """beta 0.5 at sigma 1 is half the squared error; the plain likelihood (beta 0) is minimised in sigma at |y - mu|;
    absent rows do not count; the weight sigma^(2 beta) carries no gradient."""
    from helix.model.denoise import charge_loss, gauss_nll_loss
    g = torch.Generator().manual_seed(0)
    y = torch.rand(64, generator=g) * 5; mu = y + 0.1 * torch.randn(64, generator=g); pos = y > 1
    assert torch.allclose(gauss_nll_loss(mu, torch.zeros(64), y, pos, 0.5), 0.5 * charge_loss(mu, y, pos, 0.0))
    e = (y - mu).abs()
    ls = torch.log(e).clone().requires_grad_(True)
    gauss_nll_loss(mu, ls, y, pos, 0.0).backward()
    assert ls.grad[pos].abs().max() < 1e-5                         # d/d(log s) [e^2 / 2 s^2 + log s] = 0 at s = e
    mu2 = mu.clone(); mu2[~pos] += 100
    assert torch.allclose(gauss_nll_loss(mu2, ls.detach(), y, pos, 0.5), gauss_nll_loss(mu, ls.detach(), y, pos, 0.5))


def test_hlgauss_targets_and_loss():
    """Targets are normalised and their mean is y (inside the range, to well below a bin width); the loss is minimised
    by the targets' own log-probabilities; absent rows do not count."""
    from helix.model.denoise import hlgauss_loss, hlgauss_targets
    edges = torch.linspace(0, 7.5, 1025); c = 0.5 * (edges[1:] + edges[:-1]); w = float(edges[1] - edges[0])
    y = torch.tensor([0.5, 2.0, 3.3333, 6.1]); pos = torch.tensor([True, True, True, False])
    t = hlgauss_targets(y, edges, 0.75 * w)
    assert torch.allclose(t.sum(-1), torch.ones(4))
    assert torch.allclose((t * c).sum(-1), y, atol=0.02 * w)
    best = hlgauss_loss(torch.log(t + 1e-12), y, pos, edges, 0.75 * w)
    other = hlgauss_loss(torch.log(torch.roll(t, 3, -1) + 1e-12), y, pos, edges, 0.75 * w)
    assert best < other
    lg = torch.log(t + 1e-12); lg2 = lg.clone(); lg2[3] = 0
    assert torch.allclose(hlgauss_loss(lg2, y, pos, edges, 0.75 * w), best)
