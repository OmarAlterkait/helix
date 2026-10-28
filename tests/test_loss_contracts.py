"""What the categorical objective COMPUTES, stated as the formula rather than a
recorded number.

These assert the MATH, deliberately, not a frozen output: the masking policy,
the model width and the bin grid are all expected to change, whereas the formula
is what has to stay true.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F                                    # noqa: E402

from helix.model.loss import bucketize_bins, losses_cat            # noqa: E402

K = 8


def _batch(n_cells=6, n_slot=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    occ = (torch.rand(n_cells, n_slot, generator=g) < 0.6)
    return dict(occ=occ.float(), tgt=torch.randn(n_cells, n_slot, generator=g),
                valid=torch.ones(n_cells, n_slot, dtype=torch.bool),
                band_id=torch.randint(0, 2, (n_cells,), generator=g), n_cells=n_cells)


EDGES = torch.linspace(-3, 3, K + 1).repeat(2, 1)


def test_bce_is_masked_only_and_is_the_bce_formula():
    """Occupancy BCE is supervised on MASKED cells only: visible occupancy is
    observed, so supervising it would leak the answer."""
    B = _batch()
    g = torch.Generator().manual_seed(1)
    occ_logit = torch.randn(6, 4, generator=g)
    logits = torch.randn(6, 4, K, generator=g)
    m = torch.zeros(6, dtype=torch.bool); m[::2] = True

    bce, _ = losses_cat(occ_logit, logits, B, m, EDGES)
    sel = m[:, None] & B["valid"]
    want = F.binary_cross_entropy_with_logits(occ_logit[sel], B["occ"][sel])
    assert torch.allclose(bce, want), "BCE is not the masked-only BCE it claims"

    # and it must actually IGNORE the visible half: perturbing visible logits
    # cannot move it.
    pert = occ_logit.clone(); pert[~m] += 5.0
    bce2, _ = losses_cat(pert, logits, B, m, EDGES)
    assert torch.allclose(bce, bce2), "visible occupancy leaked into the BCE"


def test_value_term_is_the_cross_entropy_of_the_true_bin_on_masked_active_slots():
    B = _batch()
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(6, 4, K, generator=g)
    m = torch.zeros(6, dtype=torch.bool); m[:3] = True

    _, val = losses_cat(torch.zeros(6, 4), logits, B, m, EDGES)
    act = m[:, None] & B["valid"] & B["occ"].bool()
    binid = bucketize_bins(B["tgt"], B["band_id"], EDGES, K)
    want = F.cross_entropy(logits[act], binid[act])
    assert torch.allclose(val, want)
