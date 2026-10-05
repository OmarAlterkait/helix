"""``band_pool``: the encoder trunk over locations (every band of a plane x wire
block x drift window pooled into one token).

Pins what the pooling must be for a comparison against band tokens to mean
anything: the default builds the original model; a location's band tokens share
one trunk feature; the result does not depend on row order; with the skip path
the per-token features still differ within a location; the trunk is trained.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from helix.model.serial import _locations                           # noqa: E402
from test_serial import SMALL, make_batch                           # noqa: E402
from test_varlen import _model                                      # noqa: E402

N = 600


def _batch():
    B = make_batch(n_cells=N)
    B["wire_pos"] = (B["wire_pos"] // 16) * 16                    # block starts, as tokenized
    B["t_phys"] = B["t_phys"].abs() * 2                           # a few locations hold several bands
    B["n_cells"] = N
    return B


def _pooled(**kw):
    m = _model(varlen=True, band_pool=1, **kw)
    m.mask_cell = (16, 128)
    return m


def test_default_builds_no_pool_parameters():
    assert not any(k.startswith("pool_") for k in _model().state_dict())
    assert {"pool_norm.weight", "pool_proj.weight", "pool_up.weight"} <= set(_pooled().state_dict())


def test_refusals():
    with pytest.raises(ValueError):
        _model(band_pool=1)                                       # needs varlen
    with pytest.raises(ValueError):
        _model(varlen=True, band_pool=SMALL["blocks"])            # no trunk left
    with pytest.raises(ValueError):
        _model(varlen=True, band_pool=1, pool_sub=(1, 1))         # one entry per band


def test_locations_and_typed_slots():
    B = _batch()
    loc, slot, lp, lt, lw = _locations(B["plane_id"], B["t_phys"], B["wire_pos"], B["band_id"], (16, 128), (1, 1, 2, 4))
    for l in range(int(loc.max()) + 1):
        r = loc == l
        assert (B["plane_id"][r] == lp[l]).all() and (B["wire_pos"][r] == lw[l]).all()
        assert ((B["t_phys"][r] - lt[l]).abs() <= 64).all()
    lo = torch.tensor([0, 1, 2, 4])[B["band_id"]]
    hi = torch.tensor([1, 2, 4, 8])[B["band_id"]]
    assert ((slot >= lo) & (slot < hi)).all()


def test_no_skip_gives_every_band_of_a_location_one_feature():
    m = _pooled(pool_skip=False).eval()
    B = _batch()
    with torch.no_grad():
        f = m.encode(B)
    loc = _locations(B["plane_id"], B["t_phys"], B["wire_pos"], B["band_id"], (16, 128), (1, 1, 2, 4))[0]
    multi = [l for l in range(int(loc.max()) + 1) if (loc == l).sum() > 1]
    assert multi
    for l in multi:
        g = f[loc == l]
        torch.testing.assert_close(g, g[:1].expand_as(g))


def test_skip_keeps_token_features_distinct_and_trains_the_trunk():
    m = _pooled().train()
    torch.nn.init.normal_(m.pool_up.weight, std=0.02)             # past its zero init
    B = _batch()
    out = m(B)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for p in (m.enc[0].qkv.weight, m.pool_proj.weight, m.enc[-1].qkv.weight, m.pool_up.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        f = m.encode(B)
    loc = _locations(B["plane_id"], B["t_phys"], B["wire_pos"], B["band_id"], (16, 128), (1, 1, 2, 4))[0]
    l = next(l for l in range(int(loc.max()) + 1) if (loc == l).sum() > 1)
    g = f[loc == l]
    assert not torch.allclose(g[0], g[1])


@pytest.mark.parametrize("skip", [True, False])
def test_row_order_does_not_matter(skip):
    m = _pooled(pool_skip=skip).eval()
    torch.nn.init.normal_(m.pool_up.weight, std=0.02)
    B = _batch()
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(1))
    Bp = {k: (v[perm] if torch.is_tensor(v) and v.shape[:1] == (N,) else v) for k, v in B.items()}
    with torch.no_grad():
        torch.testing.assert_close(m.encode(Bp), m.encode(B)[perm], rtol=1e-4, atol=1e-4)


def test_location_masked_training_decodes_band_tokens():
    m = _pooled().train()
    m.mask_mode = "location"
    B = _batch()
    mask = m.make_mask(B)
    feat, rows = m.forward_feat(B, mask, masked_only=True)
    assert feat.shape == (int(mask.sum()), SMALL["d"]) and bool(mask[rows].all())
    m2 = _pooled(pool_skip=False).train()
    feat2, _ = m2.forward_feat(B, mask, masked_only=True)
    assert torch.isfinite(feat2).all()


def test_pool_weights_are_mup_hidden():
    m = _model(varlen=True, band_pool=1, mup=True, d_base=16)
    cats = m._mup_categories()
    assert cats["pool_proj.weight"] == cats["pool_up.weight"] == "hidden"


@pytest.mark.parametrize("pw", [8, 16])
def test_tokenizer_grid_fills_each_typed_slot_once(pw):
    """Every token the tokenizer can emit (grid_center times, its per-band delays
    and per-plane offsets) lands in its own (location, sub-slot): the typed pool
    never sums two tokens into one slot, at pw16 or pw8."""
    import numpy as np
    from helix.model.tokenize import PatchConfig
    cfg = PatchConfig(pw=pw, cell_t="grid_center")
    dec = 2.0 ** np.array(cfg.lev)
    rows = []
    for g in range(6):
        for b in range(4):
            n_tb = int(4096 / (cfg.pt * dec[b]))
            tb = np.arange(n_tb)
            t = (tb * cfg.pt + cfg.pt / 2 + cfg.delta[b]) * dec[b] - cfg.toff[g % 3]
            for wb in range(3):
                rows.append(np.stack([np.full(n_tb, g), np.full(n_tb, b), np.full(n_tb, wb * pw), t], 1))
    r = torch.tensor(np.concatenate(rows))
    loc, slot, *_ = _locations(r[:, 0].long(), r[:, 3].float(), r[:, 2].float(), r[:, 1].long(), (pw, 128), (1, 1, 2, 4))
    pairs = loc * 8 + slot
    assert pairs.unique().numel() == pairs.numel()
