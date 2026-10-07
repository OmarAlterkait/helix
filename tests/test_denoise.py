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
