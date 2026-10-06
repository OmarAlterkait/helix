"""Objective options: ``vis_frac`` (visible tokens decoded) and ``noisy_target``.

Pins: the defaults are the original objective; visible rows are decoded only in
training, get the value loss and no occupancy loss; a noisy-target tokenizer
emits the normalised input as its target and ignores the clean modality.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from helix.model.head import cat_head_sparse                        # noqa: E402
from helix.model.tokenize import PatchConfig, assemble              # noqa: E402
from test_serial import SMALL, make_batch                           # noqa: E402
from test_tokenize import LENS_T, _rows                             # noqa: E402
from test_varlen import EXACT, _exact_mask, _model                  # noqa: E402


@pytest.mark.parametrize("varlen", [False, True])
def test_vis_frac_zero_is_the_original_path(varlen):
    a, b = _model(varlen=varlen).train(), _model(varlen=varlen).train()
    b.load_state_dict(a.state_dict()); b.vis_frac = 0.0
    B, m = make_batch(**EXACT), _exact_mask(EXACT["n_cells"])
    torch.manual_seed(0); fa, ra = a.forward_feat(B, m, masked_only=True)
    torch.manual_seed(0); fb, rb = b.forward_feat(B, m, masked_only=True)
    assert torch.equal(ra, rb) and torch.equal(fa, fb)


@pytest.mark.parametrize("varlen", [False, True])
def test_vis_frac_decodes_a_share_of_visible_rows_in_training_only(varlen):
    m_ = _model(varlen=varlen).train(); m_.vis_frac = 1 / 3
    B, mask = make_batch(**EXACT), _exact_mask(EXACT["n_cells"])     # 96 visible, 192 masked
    feat, rows = m_.forward_feat(B, mask, masked_only=True)
    vis = ~mask[rows]
    assert int(vis.sum()) == 32 and int((~vis).sum()) == 192
    assert rows.unique().numel() == rows.numel() and feat.shape[0] == rows.numel()
    m_.eval()
    with torch.no_grad():
        _, rows_eval = m_.forward_feat(B, mask, masked_only=True)
    assert bool(mask[rows_eval].all()) and rows_eval.numel() == 192


def test_visible_rows_get_value_loss_but_no_occupancy_loss():
    m_ = _model()
    B = make_batch(**EXACT)
    rows = torch.tensor([3, 10, 40, 41])
    feat = torch.randn(4, SMALL["d"])
    no_bce = torch.tensor([False, False, True, True])
    bce, val = cat_head_sparse(m_, feat, B, rows, no_bce=no_bce)
    bce_m, _ = cat_head_sparse(m_, feat[:2], B, rows[:2])
    _, val_all = cat_head_sparse(m_, feat, B, rows)
    torch.testing.assert_close(bce, bce_m)                 # occupancy: masked rows only
    torch.testing.assert_close(val, val_all)               # value: every decoded row


@pytest.mark.parametrize("varlen", [False, True])
def test_vis_frac_trains_end_to_end(varlen):
    m_ = _model(varlen=varlen).train(); m_.vis_frac = 1 / 3
    out = m_(make_batch(**EXACT))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert m_.dec[0].q.weight.grad.abs().sum() > 0


def test_noisy_target_is_the_normalised_input():
    gids = np.array([0, 1, 2, 3, 4, 5])
    band, gid, wire, tau, raw, raw_clean, sigma = _rows()
    kw = dict(gids=gids, n_wires=np.full(len(gids), 1969, np.int64), band_lengths=LENS_T,
              norm_sigma=sigma, cfg=PatchConfig(cell_t="grid_center"))
    clean = assemble(band, gid, wire, tau, raw, value_clean=raw_clean, **kw)
    noisy = assemble(band, gid, wire, tau, raw, value_clean=raw_clean, noisy_target=True, **kw)
    occ = noisy["occ"].astype(bool)
    np.testing.assert_array_equal(noisy["tgt"][occ], noisy["inp"][occ])
    assert not np.allclose(clean["tgt"][occ], clean["inp"][occ])   # the default stays clean
    np.testing.assert_array_equal(noisy["inp"], clean["inp"])
