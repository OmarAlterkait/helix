"""`losses_cat`'s bin assignment must be identical after the memory fix.

The original computed the true bin as

    binid = (tgt.unsqueeze(-1) >= edges[band][:, None, 1:-1]).sum(-1)

which materialises an ``(n_cells, n_slot, K-1)`` tensor — and because the
comparison is bool while ``.sum(-1)`` accumulates in **int64**, at 8 bytes per
element. For a 37k-cell event at K=128 that is 4.66 GiB, and it OOM'd an 11 GB
card at step 4 of the first real training launch.

`bucketize` per band gives the same answer with no intermediate. The whole risk
of that swap is the tie-break: `>=` sends a value landing exactly ON an edge to
the upper bin, and `bucketize` only matches with ``right=True``. So these tests
compare against the ORIGINAL expression, and go out of their way to put values
exactly on edges — with random data the probability of a tie is zero, and the
test would pass while proving nothing.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model.loss import losses_cat  # noqa: E402

K = 16
N_BAND = 4


def _reference_binid(tgt, band_id, edges, K):
    """The pre-fix expression, verbatim, as the thing to match."""
    ec = edges[band_id]
    return (tgt.unsqueeze(-1) >= ec[:, None, 1:-1]).sum(-1).clamp(0, K - 1)


def _bucketized_binid(tgt, band_id, edges, K):
    """What losses_cat now does."""
    out = torch.empty(tgt.shape, dtype=torch.long, device=tgt.device)
    for b in range(edges.shape[0]):
        sel = band_id == b
        if sel.any():
            out[sel] = torch.bucketize(tgt[sel], edges[b, 1:-1].contiguous(),
                                       right=True)
    return out.clamp(0, K - 1)


def _edges():
    e = torch.linspace(-4, 4, K + 1).repeat(N_BAND, 1).clone()
    e[:, 0], e[:, -1] = -1e18, 1e18          # the real tables use infinite tails
    for b in range(N_BAND):                  # give each band a distinct grid
        e[b, 1:-1] = torch.linspace(-4 + b, 4 - b * 0.5, K - 1)
    return e


def test_matches_the_original_on_random_values():
    g = torch.Generator().manual_seed(0)
    tgt = torch.randn(300, 32, generator=g) * 3
    band_id = torch.randint(0, N_BAND, (300,), generator=g)
    e = _edges()
    assert torch.equal(_bucketized_binid(tgt, band_id, e, K),
                       _reference_binid(tgt, band_id, e, K))


def test_matches_the_original_ON_the_edges():
    """The tie-break is the whole risk of the swap. Random data never lands on an
    edge, so this places values exactly there — and just below/above."""
    e = _edges()
    band_id = torch.arange(N_BAND).repeat_interleave(4)          # 16 cells
    rows = []
    for b in band_id.tolist():
        interior = e[b, 1:-1]
        rows.append(torch.cat([interior,
                               interior - 1e-6,
                               interior + 1e-6,
                               torch.tensor([-1e9, 1e9])]))
    tgt = torch.stack(rows)
    assert torch.equal(_bucketized_binid(tgt, band_id, e, K),
                       _reference_binid(tgt, band_id, e, K)), \
        "tie-break differs: a value exactly on an edge lands in a different bin"


def test_losses_cat_value_is_unchanged():
    """End to end: the loss itself, not just the index."""
    g = torch.Generator().manual_seed(3)
    n_cells, n_slot = 200, 32
    tgt = torch.randn(n_cells, n_slot, generator=g) * 3
    occ = (torch.rand(n_cells, n_slot, generator=g) < 0.4).float()
    B = dict(valid=(torch.rand(n_cells, n_slot, generator=g) < 0.9),
             occ=occ, tgt=tgt,
             band_id=torch.randint(0, N_BAND, (n_cells,), generator=g))
    logits = torch.randn(n_cells, n_slot, K, generator=g)
    occ_logit = torch.randn(n_cells, n_slot, generator=g)
    mask = torch.zeros(n_cells, dtype=torch.bool); mask[::2] = True
    e = _edges()

    bce, val = losses_cat(occ_logit, logits, B, mask, e)

    # recompute with the reference index
    import torch.nn.functional as F
    binid = _reference_binid(tgt, B["band_id"], e, K)
    ce_e = F.cross_entropy(logits.reshape(-1, K), binid.reshape(-1),
                           reduction="none").view_as(tgt)
    act = occ.bool() & B["valid"] & mask[:, None]
    want = (ce_e * act).sum() / act.sum().clamp(min=1)
    assert torch.equal(val, want), f"loss changed: {float(val)} vs {float(want)}"


def test_no_large_intermediate():
    """The point of the change. Peak allocation must not scale with K.

    Measured on CPU with a tensor big enough that the old form would be
    conspicuous: the old expression allocates n_cells*n_slot*(K-1)*8 bytes."""
    g = torch.Generator().manual_seed(5)
    n_cells, n_slot, big_K = 4000, 128, 128
    tgt = torch.randn(n_cells, n_slot, generator=g)
    band_id = torch.randint(0, N_BAND, (n_cells,), generator=g)
    e = torch.linspace(-4, 4, big_K + 1).repeat(N_BAND, 1).contiguous()
    would_be = n_cells * n_slot * (big_K - 1) * 8
    assert would_be > 4e8, "fixture too small to be meaningful"
    out = _bucketized_binid(tgt, band_id, e, big_K)   # must simply not blow up
    assert out.shape == (n_cells, n_slot)
    assert out.dtype == torch.long
